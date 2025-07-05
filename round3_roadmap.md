# Round 3

## Algorithm challenge

Our inhabitants really like volcanic rock. So much even, that they invented a new tradable good, `VOLCANIC ROCK VOUCHERS`. The vouchers will give you the right but not obligation to buy `VOLCANIC ROCK` at a certain price (strike price) at voucher expiry timestamp. These vouchers can be traded as a separate item on the island’s exchange. Of course you will have to pay a premium for these vouchers, but if your strategy is solid as a rock, SeaShells spoils will be waiting for you on the horizon. 

There are five Volcanic Rock Vouchers, each with their own **strike price** and **premium.** 

**At beginning of Round 1, all the Vouchers have 7 trading days to expire. By end of Round 5, vouchers will have 2 trading days left to expire.**

Position limits for the newly introduced products:

- `VOLCANIC_ROCK`: 400

`VOLCANIC_ROCK_VOUCHER_9500` :

- Position Limit: 200
- Strike Price: 9,500 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_9750` :

- Position Limit: 200
- Strike Price: 9,750 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10000` :

- Position Limit: 200
- Strike Price: 10,000 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10250` :

- Position Limit: 200
- Strike Price: 10,250 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

`VOLCANIC_ROCK_VOUCHER_10500` :

- Position Limit: 200
- Strike Price: 10,500 SeaShells
- Expiration deadline: 7 days (1 round = 1 day) starting from round 1

### Hint for Algorithmic Challenge

Hello everyone, hope you're enjoying the VOLCANIC_ROCK vouchers and a variety of trading strategies these new products introduce. While digging for the rock, Archipelago residents found some ancient mathematics sharing insights into VOLCANIC_ROCK voucher trading. Here's what the message with obscure and advanced mathematics read,

Message begins,

I have discovered a strategy which will make ArchiCapital the biggest trading company ever. Here's how my thesis goes,

t: Timestamp
St: Voucher Underlying Price at t
K: Strike
TTE: Remaining Time till expiry at t
Vt: Voucher price of strike K at t

Compute,

m_t = log(K/St)/ sqrt(TTE)
v_t = BlackScholes ImpliedVol(St, Vt, K, TTE)

for each t, plot v_t vs m_t and fit a parabolic curve to filter random noise.

This fitted v_t(m_t) allows me to evaluate opportunities between different strikes. I also call fitted v_t(m_t=0) the base IV and I have identified interesting patterns in timeseries of base IV.

Message ends.

## Manual challenge

A big group of Sea Turtles is visiting our shores, bringing with them an opportunity to acquire some top grade `FLIPPERS`. You only have two chances to offer a good price. Each one of the Sea Turtles will accept the lowest bid that is over their reserve price. 

The distribution of reserve prices is uniform between 160–200 and 250–320, but none of the Sea Turtles will trade between 200 and 250 due to some ancient superstition.

For your second bid, they also take into account the average of the second bids by other traders in the archipelago. They’ll trade with you when your offer is above the average of all second bids. But if you end up under the average, the probability of a deal decreases rapidly. 

To simulate this probability, the PNL obtained from trading with a fish for which your second bid is under the average of all second bids will be scaled by a factor *p*:

$$
p = (\frac{320 – \text{average bid}}{320 – \text{your bid}})^3
$$

You know there’s a constant desire for Flippers on the archipelago. So, at the end of the round, you’ll be able to sell them for 320 SeaShells ****a piece.

Think hard about how you want to set your two bids, place your feet firmly in the sand and brace yourself, because this could get messy.
