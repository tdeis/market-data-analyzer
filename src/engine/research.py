from src.analytics.reporter import ResearchReporter


class AutoResearchEngine:

    def __init__(self, returns, strategy_name="strategy"):

        self.returns = returns
        self.name = strategy_name
        self.reporter = ResearchReporter(returns)

    # =========================
    # META SUMMARY (HEDGE FUND STYLE)
    # =========================
    def meta_summary(self):

        sharpe = self.reporter.sharpe()
        dd = self.reporter.max_drawdown()

        if sharpe > 1.5 and dd > -0.2:
            return "High-quality strategy with strong risk-adjusted returns and controlled drawdowns."
        elif sharpe > 1:
            return "Moderate quality strategy with usable but inconsistent alpha."
        elif sharpe > 0:
            return "Weak edge detected; strategy may require feature or regime improvements."
        else:
            return "No meaningful alpha detected; strategy likely non-viable."

    # =========================
    # FULL AUTO REPORT TEXT
    # =========================
    def generate_narrative(self):

        r = self.reporter

        sharpe = r.sharpe()
        dd = r.max_drawdown()
        vol = r.volatility()

        insights = r.generate_insights()

        narrative = []

        narrative.append(f"Strategy: {self.name}")
        narrative.append("")
        narrative.append("EXECUTIVE SUMMARY")
        narrative.append(self.meta_summary())
        narrative.append("")

        narrative.append("KEY METRICS")
        narrative.append(f"- Sharpe Ratio: {sharpe:.2f}")
        narrative.append(f"- Max Drawdown: {dd:.2%}")
        narrative.append(f"- Volatility: {vol:.2%}")
        narrative.append("")

        narrative.append("MODEL DIAGNOSTICS")
        for i in insights:
            narrative.append(f"- {i}")

        narrative.append("")
        narrative.append("RESEARCH INTERPRETATION")
        narrative.append(r._interpret_strategy())

        return "\n".join(narrative)
