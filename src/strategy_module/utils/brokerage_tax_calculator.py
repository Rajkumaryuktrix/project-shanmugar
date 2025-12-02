from dataclasses import dataclass, asdict

class IndianTradeCostCalculator:
    """
    Cost engine for NSE/BSE equities, futures & options (May 2025 schedule).
    Override broker constants via __init__ if yours differ.
    """

    # ─────────────────────────────────────  Default constants  ───────────────────────────────────── #
    BROKER_RATE   = 0.0003        # 0.03 % of turnover
    BROKER_CAP    = 20            # ₹ cap per order
    DELIVERY_BROKERAGE = 0.0      # most discount brokers charge zero on delivery

    STT_RATES = {
        'delivery':       0.001,      # 0.10 % on buy & sell
        'intraday_sell':  0.00025,    # 0.025 % on sell only
        'futures_sell':   0.000125,   # 0.0125 %
        'options_sell':   0.000625,   # 0.0625 % of premium
    }

    # Minimum STT amounts
    STT_MIN = {
        'delivery':       1.0,        # Minimum ₹1
        'intraday_sell':  0.5,        # Minimum ₹0.5
        'futures_sell':   0.5,        # Minimum ₹0.5
        'options_sell':   0.5,        # Minimum ₹0.5
    }

    NSE_TXN = {                     # Exchange transaction charges
        'cash': 0.0000297,
        'fut' : 0.0000173,
        'opt' : 0.0003503,          # on premium
    }

    # Minimum exchange transaction charges
    NSE_TXN_MIN = {
        'cash': 0.5,                # Minimum ₹0.5
        'fut' : 0.5,                # Minimum ₹0.5
        'opt' : 0.5,                # Minimum ₹0.5
    }

    SEBI_RATE = 0.000001           # 0.0001 %
    SEBI_MIN = 0.5                 # Minimum ₹0.5
    GST       = 0.18               # 18 %
    
    STAMP = {
        'delivery': 0.00015,
        'intraday': 0.00003,
        'futures' : 0.00002,
        'options' : 0.00003,        # on premium
    }
    
    # Minimum stamp duty amounts
    STAMP_MIN = {
        'delivery': 1.0,            # Minimum ₹1
        'intraday': 1.0,            # Minimum ₹1
        'futures' : 1.0,            # Minimum ₹1
        'options' : 1.0,            # Minimum ₹1
    }
    
    DP_CHARGE = 13 * 1.18          # ₹13 + 18 % GST, per scrip sell (delivery)

    # ──────────────────────────────────────  Trade model  ────────────────────────────────────────── #
    @dataclass
    class Trade:
        segment: str               # 'delivery' | 'intraday' | 'futures' | 'options'
        buy_value: float           # ₹ (price × qty) for the buy leg
        sell_value: float          # ₹ (price × qty) for the sell leg
        premium_turnover: float = 0.0  # options: premium on SELL (price × lot × lots)

    # ───────────────────────────────────────  Init  ─────────────────────────────────────────────── #
    def __init__(self, *, broker_rate=None, broker_cap=None, delivery_brokerage=None):
        if broker_rate       is not None: self.BROKER_RATE        = broker_rate
        if broker_cap        is not None: self.BROKER_CAP         = broker_cap
        if delivery_brokerage is not None: self.DELIVERY_BROKERAGE = delivery_brokerage

    # ──────────────────────────────────────  Internals  ─────────────────────────────────────────── #
    def _brokerage(self, buy: float, sell: float, delivery: bool) -> float:
        if delivery:
            return self.DELIVERY_BROKERAGE
        buy_bro  = min(buy  * self.BROKER_RATE, self.BROKER_CAP)
        sell_bro = min(sell * self.BROKER_RATE, self.BROKER_CAP)
        return buy_bro + sell_bro

    def _stt(self, trade: "IndianTradeCostCalculator.Trade") -> float:
        """Calculate STT with minimum charge validation"""
        if trade.segment == 'delivery':
            stt = self.STT_RATES['delivery'] * (trade.buy_value + trade.sell_value)
            return max(stt, self.STT_MIN['delivery'])
        elif trade.segment == 'intraday':
            stt = self.STT_RATES['intraday_sell'] * trade.sell_value
            return max(stt, self.STT_MIN['intraday_sell'])
        elif trade.segment == 'futures':
            stt = self.STT_RATES['futures_sell'] * trade.sell_value
            return max(stt, self.STT_MIN['futures_sell'])
        else:  # options
            stt = self.STT_RATES['options_sell'] * trade.premium_turnover
            return max(stt, self.STT_MIN['options_sell'])

    def _exchange_charges(self, trade: "IndianTradeCostCalculator.Trade") -> float:
        """Calculate exchange transaction charges with minimum charge validation"""
        if trade.segment in ('delivery', 'intraday'):
            exch = (trade.buy_value + trade.sell_value) * self.NSE_TXN['cash']
            return max(exch, self.NSE_TXN_MIN['cash'])
        elif trade.segment == 'futures':
            exch = (trade.buy_value + trade.sell_value) * self.NSE_TXN['fut']
            return max(exch, self.NSE_TXN_MIN['fut'])
        else:  # options
            exch = trade.premium_turnover * self.NSE_TXN['opt']
            return max(exch, self.NSE_TXN_MIN['opt'])

    def _stamp_duty(self, trade: "IndianTradeCostCalculator.Trade") -> float:
        """Calculate stamp duty with minimum charge validation"""
        stamp = self.STAMP[trade.segment] * trade.buy_value
        return max(stamp, self.STAMP_MIN[trade.segment])

    # ────────────────────────────────────  Public API  ──────────────────────────────────────────── #
    def calc_charges(self, trade: "IndianTradeCostCalculator.Trade") -> dict:
        """
        Returns a dict with the full fee breakdown + total_cost.
        """
        turnover = trade.buy_value + trade.sell_value
        delivery = trade.segment == 'delivery'

        # Brokerage
        bro = self._brokerage(trade.buy_value, trade.sell_value, delivery)

        # STT / CTT with minimum charge validation
        stt = self._stt(trade)

        # Exchange transaction charge with minimum charge validation
        exch = self._exchange_charges(trade)

        # SEBI fee + GST on SEBI fee
        sebi = max(turnover * self.SEBI_RATE, self.SEBI_MIN)
        gst_sebi = sebi * self.GST

        # GST on (brokerage + exchange)
        gst_other = (bro + exch) * self.GST
        gst = gst_other + gst_sebi

        # Stamp duty with minimum charge validation
        stamp = self._stamp_duty(trade)

        # DP charges on delivery sells
        dp = self.DP_CHARGE if (delivery and trade.sell_value > 0) else 0.0

        total = bro + stt + exch + sebi + gst + stamp + dp
        return {
            'brokerage'   : bro,
            'stt_ctt'     : stt,
            'exchange'    : exch,
            'sebi'        : sebi,
            'gst'         : gst,
            'stamp_duty'  : stamp,
            'dp_charges'  : dp,
            'total_cost'  : total,
            'inputs'      : asdict(trade),   # handy for audit
        }


# ───────────────────────────── Example usage ─────────────────────────────
if __name__ == "__main__":
    calc = IndianTradeCostCalculator()          # or override caps/rates here
    trade = calc.Trade(segment='intraday', buy_value=50_000, sell_value=51_000)
    print(calc.calc_charges(trade))
