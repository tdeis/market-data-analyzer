from orchestrator import run_research


def main():

    output = run_research(["AAPL", "TSLA", "SPY"])

    print(output["leaderboard"])
    print(output["db_leaderboard"])


if __name__ == "__main__":
    main()
