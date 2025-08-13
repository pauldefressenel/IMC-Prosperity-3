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
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
          Product.RAINFOREST_RESIN: 50, 
          Product.KELP: 50, 
          Product.PICNIC_BASKET1 : 60, 
          Product.PICNIC_BASKET2 : 100, 
          Product.CROISSANTS : 250, 
          Product.JAMS : 350, 
          Product.DJEMBES : 60
        }

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



        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        return result, conversions, traderData
