from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
from typing import Dict, Any
from typing import Any, Dict, List, Optional


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SQUID_INK="SQUID_INK"
    SPREAD="SPREAD"
    VOLCANIC_ROCK="VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.28,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SPREAD: {
        "spread_std_window": 15,
        "zscore_threshold":2
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "strike": 10000,
        "vol_window": 70,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.PICNIC_BASKET1 : 60, Product.PICNIC_BASKET2 : 100, Product.CROISSANTS : 250, Product.JAMS : 350, Product.DJEMBES : 60, Product.VOLCANIC_ROCK:400, Product.VOLCANIC_ROCK_VOUCHER_10000:200}

        # SQUID_INK params for Round 2
        self.squid_limit = 50
        self.squid_std_window = 20
        self.squid_mid_price_history = []
        self.squid_default_threshold = 5
        self.squid_threshold_warmup_count = 0
        self.squid_required_threshold_warmup = 20
        self.squid_ema_alpha = 0.005
        self.squid_ema_value = None
        self.squid_base_value = 1900
        self.squid_warmup_count = 0
        self.squid_required_warmup = 100

        #base IV params
        # Constants for the Black–Scholes calculations.
        # -----------------------------------------------------------------------------
        self._p = 0.3275911
        self._a1 = 0.254829592
        self._a2 = -0.284496736
        self._a3 = 1.421413741
        self._a4 = -1.453152027
        self._a5 = 1.061405429
        self._sqrt2 = math.sqrt(2.0)
    
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def update_squid_ema(self, price: float) -> None:
        if self.squid_ema_value is None:
            self.squid_ema_value = price
        else:
            self.squid_ema_value = self.squid_ema_alpha * price + (1 - self.squid_ema_alpha) * self.squid_ema_value
        self.squid_warmup_count += 1

    def compute_squid_threshold(self) -> float:
        if self.squid_threshold_warmup_count < self.squid_required_threshold_warmup or len(self.squid_mid_price_history) < self.squid_std_window:
            return self.squid_default_threshold
        mean_price = sum(self.squid_mid_price_history) / len(self.squid_mid_price_history)
        variance = sum((p - mean_price) ** 2 for p in self.squid_mid_price_history) / len(self.squid_mid_price_history)
        return variance * 7

    def generate_squid_orders(self, order_depth: OrderDepth, position: int, threshold: float) -> List[Order]:
        orders: List[Order] = []
        fair_value = self.squid_ema_value if self.squid_ema_value is not None else self.squid_base_value

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask < fair_value - threshold:
                quantity = min(-order_depth.sell_orders[best_ask], self.squid_limit - position)
                if quantity > 0:
                    orders.append(Order(Product.SQUID_INK, best_ask, quantity))

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid > fair_value + threshold:
                quantity = min(order_depth.buy_orders[best_bid], self.squid_limit + position)
                if quantity > 0:
                    orders.append(Order(Product.SQUID_INK, best_bid, -quantity))

        return orders


    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def trade_custom_basket_spread(self, state:TradingState, spread_data: Dict[str, Any], order_depths: Dict[str, OrderDepth]):
        od=order_depths
        pos = state.position

        required = [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2, Product.DJEMBES]
        if not all(p in order_depths for p in required):
            return {}

        mid_b1 = self.get_swmid(order_depths[Product.PICNIC_BASKET1])
        mid_b2 = self.get_swmid(order_depths[Product.PICNIC_BASKET2])
        mid_dj =self.get_swmid(order_depths[Product.DJEMBES])

        if None in (mid_b1, mid_b2, mid_dj):
            return {}

        spread = mid_b1 - 1.5 * mid_b2 - mid_dj

        # Track spread history
        spread_data["history"].append(spread)

        window = self.params[Product.SPREAD]["spread_std_window"]
        if len(spread_data["history"]) > window:
            spread_data["history"].pop(0)

        if len(spread_data["history"]) < window:
            return {}

        rolling_mean = np.mean(spread_data["history"])
        rolling_std = np.std(spread_data["history"])
        if rolling_std == 0:
            return {}

        zscore = (spread - rolling_mean) / rolling_std
        spread_data["prev_zscore"] = zscore
        spread_data["last_spread"] = spread

        threshold = self.params[Product.SPREAD]["zscore_threshold"]
        unwind_zone = 0.5  # Z-score threshold for mean-reversion
        unit = 5          # Trade size per signal
        result = {p: [] for p in required}

        # Estimate current synthetic position
        b1_pos = pos.get(Product.PICNIC_BASKET1, 0)
        b2_pos = pos.get(Product.PICNIC_BASKET2, 0)
        dj_pos = pos.get(Product.DJEMBES, 0)

        # OPEN positions when spread diverges
        if zscore > threshold and b1_pos ==0 and b2_pos==0 and dj_pos==0:
            result[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, min(od[Product.PICNIC_BASKET1].buy_orders), -unit))
            result[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, max(od[Product.PICNIC_BASKET2].sell_orders), int(unit * 1.5)))
            result[Product.DJEMBES].append(Order(Product.DJEMBES, max(od[Product.DJEMBES].sell_orders), unit))

        elif zscore < -threshold and b1_pos ==0 and b2_pos==0 and dj_pos==0:
            result[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, max(od[Product.PICNIC_BASKET1].sell_orders), unit))
            result[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, min(od[Product.PICNIC_BASKET2].buy_orders), -int(unit * 1.5)))
            result[Product.DJEMBES].append(Order(Product.DJEMBES, min(od[Product.DJEMBES].buy_orders), -unit))

        # CLOSE when mean-reverting
        elif abs(zscore) < unwind_zone:
            if b1_pos > 0:
                direction = 1
                result[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1,
                                                            max(od[Product.PICNIC_BASKET1].sell_orders) if direction > 0 else min(od[Product.PICNIC_BASKET1].buy_orders),
                                                            unit * direction))
            if b2_pos <0:
                direction=1
                result[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2,
                                                            min(od[Product.PICNIC_BASKET2].buy_orders) if direction > 0 else max(od[Product.PICNIC_BASKET2].sell_orders),
                                                            -int(unit * 1.5 * direction)))
            if dj_pos<0:
                direction = 1
                result[Product.DJEMBES].append(Order(Product.DJEMBES,
                                                    min(od[Product.DJEMBES].buy_orders) if direction > 0 else max(od[Product.DJEMBES].sell_orders),
                                                    -unit * direction))
            if b1_pos < 0:
                direction = -1
                result[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1,
                                                            max(od[Product.PICNIC_BASKET1].sell_orders) if direction > 0 else min(od[Product.PICNIC_BASKET1].buy_orders),
                                                            unit * direction))
            if b2_pos >0:
                direction=-1
                result[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2,
                                                            min(od[Product.PICNIC_BASKET2].buy_orders) if direction > 0 else max(od[Product.PICNIC_BASKET2].sell_orders),
                                                            -int(unit * 1.5 * direction)))
            if dj_pos>0:
                direction = -1
                result[Product.DJEMBES].append(Order(Product.DJEMBES,
                                                    min(od[Product.DJEMBES].buy_orders) if direction > 0 else max(od[Product.DJEMBES].sell_orders),
                                                    -unit * direction))

        return result

    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def normal_pdf(x, mu=0, sigma=1):
        if sigma <= 0:
            raise ValueError("Standard deviation sigma must be positive.")
        coeff = 1 / (math.sqrt(2 * math.pi) * sigma)
        exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
        return coeff * math.exp(exponent)


    def estimate_call_delta(self, spot: float, strike: float, iv: float, time: float = 1.0) -> float:
        if spot <= 0 or iv <= 0 or time <= 0:
            return 0.0
        d1 = (math.log(spot / strike) + 0.5 * iv**2 * time) / (iv * math.sqrt(time))
        return self.norm_cdf(d1)

    def implied_volatility_call(self,premium, S, K, T, r=0.0):
        if S <= 0 or K <= 0 or T <= 0 or premium <= 0:
            return np.nan
        try:
            # Initial guess
            sigma = 0.2
            for i in range(100):
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                price = S * self.norm_cdf(d1) - K * np.exp(-r * T) * self.norm_cdf(d2)
                vega = S * self.norm_pdf(d1) * np.sqrt(T)
                if vega == 0:
                    return np.nan
                sigma -= (price - premium) / vega
                if sigma < 0:
                    return np.nan
            return sigma
        except:
            return np.nan

    def find_atm_voucher(voucher_strikes: dict, underlying_price: float) -> str:
    
        return min(voucher_strikes.items(), key=lambda item: abs(item[1] - underlying_price))[0]

    def base_iv_strat(self, state):
        od = state.order_depths
        products = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        underlying_price = self.get_swmid(state.order_depths["VOLCANIC_ROCK"])
        if underlying_price is None:
            return {}

        prod = "VOLCANIC_ROCK_VOUCHER_10000"
        orders={}
        max_pos = 200
        pos = state.position.get(prod, 0)
        if pos <= max_pos:
            ask_prices = od[prod].sell_orders
            if ask_prices:
                best_ask = min(ask_prices)
                qty = min(200, od[prod].sell_orders[best_ask])
                orders[prod] = [Order(prod, best_ask, qty)]
                self.last_base_iv_trade = self.implied_volatility_call(best_ask, underlying_price, 10000, 5/252,r=0.0)
                self.option_delta=self.estimate_call_delta(best_ask, 10000, self.last_base_iv_trade, 5/252)
        else:
            bid_prices = od[prod].buy_orders
            if bid_prices:
                best_bid = max(bid_prices)
                current_iv = self.implied_volatility_call(best_bid, underlying_price, 10000, 252/5, r=0.0)
                if current_iv - self.last_base_iv_trade < -0.01:
                    qty = min(200, od[prod].buy_orders[best_bid])
                    orders[prod] = [Order(prod, best_bid, -qty)]

        option_position = state.position.get("VOLCANIC_ROCK_VOUCHER_10000", 0)
        od = state.order_depths["VOLCANIC_ROCK_VOUCHER_10000"]
        if od.buy_orders or od.sell_orders:
            best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
            if best_bid is not None and best_ask is not None:
                premium = (best_bid + best_ask) / 2
            elif best_bid is not None:
                premium = best_bid
            elif best_ask is not None:
                premium = best_ask
            else:
                return {}  # still no usable price
        od_und = state.order_depths["VOLCANIC_ROCK"]
        if od_und.buy_orders or od_und.sell_orders:
            best_bid = max(od_und.buy_orders.keys()) if od_und.buy_orders else None
            best_ask = min(od_und.sell_orders.keys()) if od_und.sell_orders else None
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid = best_bid
            elif best_ask is not None:
                mid = best_ask
            else:
                return {}  # still no usable price
        current_iv = self.implied_volatility_call(premium, mid, 10000, 5/252, r=0.0)
        current_delta = self.estimate_call_delta(mid,10000, current_iv, 5/252)
        delta_pos = (self.option_delta-current_delta)*option_position
        if self.option_delta-current_delta >0.5:
            qty = delta_pos
            orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -qty)]
        elif self.option_delta-current_delta < -0.5:
            qty=delta_pos
            orders["VOLCANIC_ROCK"]=[Order("VOLCANIC_ROCK", best_ask, qty)]

        return orders
                

        

    def get_mid(self, order_depth):
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        elif bids:
            return max(bids)
        elif asks:
            return min(asks)
        return None



    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_resin_position,
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        # SQUID_INK logic
        if Product.SQUID_INK in state.order_depths:
            squid_position = state.position.get(Product.SQUID_INK, 0)
            squid_od = state.order_depths[Product.SQUID_INK]

            if squid_od.buy_orders and squid_od.sell_orders:
                best_bid = max(squid_od.buy_orders.keys())
                best_ask = min(squid_od.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2.0
            elif squid_od.buy_orders:
                mid_price = max(squid_od.buy_orders.keys())
            elif squid_od.sell_orders:
                mid_price = min(squid_od.sell_orders.keys())
            else:
                mid_price = self.squid_ema_value if self.squid_ema_value is not None else self.squid_base_value

            self.squid_mid_price_history.append(mid_price)
            if len(self.squid_mid_price_history) > self.squid_std_window:
                self.squid_mid_price_history.pop(0)

            if self.squid_threshold_warmup_count < self.squid_required_threshold_warmup:
                self.squid_threshold_warmup_count += 1

            threshold = self.compute_squid_threshold()
            self.update_squid_ema(mid_price)
            fair_val = self.squid_ema_value

            squid_orders = self.generate_squid_orders(squid_od, squid_position, threshold) if self.squid_warmup_count >= self.squid_required_warmup else []
            result[Product.SQUID_INK] = squid_orders

        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        

        spread_orders = self.trade_custom_basket_spread(state=state,spread_data=traderObject[Product.SPREAD],order_depths=state.order_depths) or {}
        for product, orders in spread_orders.items():
            result[product] = result.get(product, []) + orders
                
            
        option_orders = self.base_iv_strat(state=state)
        for product, orders in option_orders.items():
            result[product]=result.get(product,[])+orders


        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        return result, conversions, traderData
