import sqlite3
import json


class ExperimentDB:

    def __init__(self):

        self.conn = sqlite3.connect("experiments.db")
        self._init()
        self._migrate()

    def _init(self):

        c = self.conn.cursor()

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT,
            sharpe REAL,
            drawdown REAL,
            volatility REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        self.conn.commit()

    def log(self, strategy, metrics, metadata):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO experiments (
                strategy,
                sharpe,
                drawdown,
                volatility,
                metadata
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                strategy,
                metrics["sharpe"],
                metrics.get("drawdown", 0.0),
                metrics["volatility"],
                json.dumps(metadata),  # ✅ FIX
            ),
        )

        self.conn.commit()

    def leaderboard(self):

        c = self.conn.cursor()

        c.execute("SELECT strategy, sharpe, drawdown, volatility FROM experiments")

        return c.fetchall()

    def _migrate(self):

        c = self.conn.cursor()

        try:
            c.execute("ALTER TABLE experiments ADD COLUMN drawdown REAL")
            self.conn.commit()
        except:
            pass  # column already exists
