"""
Economic and Tokenization Layer Module

Multi-layered revenue engines with tokenized, recursive economic flows
for self-sustaining growth and long-term economic impact forecasting.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TokenType(Enum):
    """Types of tokens in the QRATUM economy."""

    UTILITY = "utility"  # Network access and computation
    GOVERNANCE = "governance"  # Voting rights
    STAKE = "stake"  # Validator staking
    DATA = "data"  # Data access rights
    COMPLIANCE = "compliance"  # Compliance verification


class RevenueType(Enum):
    """Types of revenue streams."""

    TRANSACTION_FEE = "transaction_fee"
    DATA_SERVICE = "data_service"
    ANALYTICS = "analytics"
    COMPLIANCE = "compliance"
    STAKING_REWARD = "staking_reward"


@dataclass(frozen=True)
class TokenFlow:
    """Represents a single token flow/transaction.

    Attributes:
        flow_id: Unique flow identifier
        token_type: Type of token
        amount: Amount of tokens
        sender: Sender address/ID
        receiver: Receiver address/ID
        timestamp: Transaction timestamp
        fee: Transaction fee
        metadata: Additional metadata
    """

    flow_id: str
    token_type: TokenType
    amount: float
    sender: str
    receiver: str
    timestamp: str
    fee: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize flow to dictionary."""
        return {
            "flow_id": self.flow_id,
            "token_type": self.token_type.value,
            "amount": self.amount,
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp,
            "fee": self.fee,
            "metadata": self.metadata,
        }


@dataclass
class RevenueStream:
    """Represents a revenue stream.

    Attributes:
        stream_id: Unique stream identifier
        revenue_type: Type of revenue
        base_rate: Base rate for revenue calculation
        volume: Current volume
        total_revenue: Total accumulated revenue
        active: Whether the stream is active
        metadata: Additional metadata
    """

    stream_id: str
    revenue_type: RevenueType
    base_rate: float = 0.001
    volume: float = 0.0
    total_revenue: float = 0.0
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_revenue(self, volume: float) -> float:
        """Calculate revenue for given volume.

        Args:
            volume: Transaction/service volume

        Returns:
            Calculated revenue
        """
        if not self.active:
            return 0.0
        return volume * self.base_rate

    def record_revenue(self, amount: float) -> float:
        """Record revenue and return updated total.

        Args:
            amount: Revenue amount

        Returns:
            Updated total revenue
        """
        self.total_revenue += amount
        self.volume += amount / self.base_rate if self.base_rate > 0 else 0
        return self.total_revenue

    def to_dict(self) -> dict[str, Any]:
        """Serialize stream to dictionary."""
        return {
            "stream_id": self.stream_id,
            "revenue_type": self.revenue_type.value,
            "base_rate": self.base_rate,
            "volume": self.volume,
            "total_revenue": self.total_revenue,
            "active": self.active,
            "metadata": self.metadata,
        }


@dataclass
class TransactionFee(RevenueStream):
    """Transaction fee revenue stream.

    Attributes:
        min_fee: Minimum transaction fee
        max_fee: Maximum transaction fee
        dynamic_rate: Whether to use dynamic pricing
    """

    min_fee: float = 0.0001
    max_fee: float = 0.01
    dynamic_rate: bool = True

    def __post_init__(self) -> None:
        self.revenue_type = RevenueType.TRANSACTION_FEE

    def calculate_fee(self, transaction_value: float, network_load: float = 0.5) -> float:
        """Calculate transaction fee based on value and network load.

        Args:
            transaction_value: Value of the transaction
            network_load: Current network load (0.0-1.0)

        Returns:
            Calculated fee
        """
        if self.dynamic_rate:
            # Dynamic fee based on network load
            load_multiplier = 1 + (network_load * 2)
            fee = transaction_value * self.base_rate * load_multiplier
        else:
            fee = transaction_value * self.base_rate

        return max(self.min_fee, min(self.max_fee, fee))


@dataclass
class DataAsAService(RevenueStream):
    """Data-as-a-Service revenue stream.

    Attributes:
        data_tiers: Pricing tiers for data access
        subscription_base: Base subscription rate
    """

    data_tiers: dict[str, float] = field(default_factory=dict)
    subscription_base: float = 100.0

    def __post_init__(self) -> None:
        self.revenue_type = RevenueType.DATA_SERVICE
        if not self.data_tiers:
            self.data_tiers = {
                "basic": 1.0,
                "standard": 2.5,
                "premium": 5.0,
                "enterprise": 10.0,
            }

    def calculate_data_access_cost(self, tier: str, queries: int) -> float:
        """Calculate cost for data access.

        Args:
            tier: Service tier
            queries: Number of queries

        Returns:
            Total cost
        """
        tier_multiplier = self.data_tiers.get(tier, 1.0)
        return self.subscription_base * tier_multiplier + (queries * self.base_rate)


@dataclass
class AnalyticsRevenue(RevenueStream):
    """Analytics service revenue stream.

    Attributes:
        model_pricing: Pricing for different analytics models
        compute_rate: Rate per compute unit
    """

    model_pricing: dict[str, float] = field(default_factory=dict)
    compute_rate: float = 0.01

    def __post_init__(self) -> None:
        self.revenue_type = RevenueType.ANALYTICS
        if not self.model_pricing:
            self.model_pricing = {
                "basic_prediction": 10.0,
                "advanced_ml": 50.0,
                "deep_learning": 200.0,
                "custom_model": 500.0,
            }

    def calculate_analytics_cost(self, model_type: str, compute_units: int) -> float:
        """Calculate cost for analytics service.

        Args:
            model_type: Type of analytics model
            compute_units: Number of compute units used

        Returns:
            Total cost
        """
        model_cost = self.model_pricing.get(model_type, 10.0)
        compute_cost = compute_units * self.compute_rate
        return model_cost + compute_cost


@dataclass
class ComplianceRevenue(RevenueStream):
    """Compliance service revenue stream.

    Attributes:
        certification_fees: Fees for different certifications
        audit_rate: Rate for audit services
    """

    certification_fees: dict[str, float] = field(default_factory=dict)
    audit_rate: float = 1000.0

    def __post_init__(self) -> None:
        self.revenue_type = RevenueType.COMPLIANCE
        if not self.certification_fees:
            self.certification_fees = {
                "basic": 500.0,
                "standard": 2000.0,
                "advanced": 5000.0,
                "enterprise": 20000.0,
            }

    def calculate_compliance_cost(self, certification_level: str, audit_hours: int) -> float:
        """Calculate cost for compliance services.

        Args:
            certification_level: Level of certification
            audit_hours: Hours of audit service

        Returns:
            Total cost
        """
        cert_fee = self.certification_fees.get(certification_level, 500.0)
        audit_cost = audit_hours * self.audit_rate
        return cert_fee + audit_cost


@dataclass
class TokenAccount:
    """Account holding multiple token types.

    Attributes:
        account_id: Unique account identifier
        balances: Dictionary of token balances by type
        staked: Dictionary of staked amounts by type
        created_at: Account creation timestamp
    """

    account_id: str
    balances: dict[TokenType, float] = field(default_factory=dict)
    staked: dict[TokenType, float] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        # Initialize all token types with zero balance
        for tt in TokenType:
            if tt not in self.balances:
                self.balances[tt] = 0.0
            if tt not in self.staked:
                self.staked[tt] = 0.0

    def deposit(self, token_type: TokenType, amount: float) -> float:
        """Deposit tokens into account.

        Args:
            token_type: Type of token
            amount: Amount to deposit

        Returns:
            New balance
        """
        self.balances[token_type] = self.balances.get(token_type, 0.0) + amount
        return self.balances[token_type]

    def withdraw(self, token_type: TokenType, amount: float) -> float | None:
        """Withdraw tokens from account.

        Args:
            token_type: Type of token
            amount: Amount to withdraw

        Returns:
            New balance or None if insufficient funds
        """
        current = self.balances.get(token_type, 0.0)
        if current >= amount:
            self.balances[token_type] = current - amount
            return self.balances[token_type]
        return None

    def stake(self, token_type: TokenType, amount: float) -> bool:
        """Stake tokens.

        Args:
            token_type: Type of token
            amount: Amount to stake

        Returns:
            True if staking successful
        """
        current = self.balances.get(token_type, 0.0)
        if current >= amount:
            self.balances[token_type] = current - amount
            self.staked[token_type] = self.staked.get(token_type, 0.0) + amount
            return True
        return False

    def unstake(self, token_type: TokenType, amount: float) -> bool:
        """Unstake tokens.

        Args:
            token_type: Type of token
            amount: Amount to unstake

        Returns:
            True if unstaking successful
        """
        current_staked = self.staked.get(token_type, 0.0)
        if current_staked >= amount:
            self.staked[token_type] = current_staked - amount
            self.balances[token_type] = self.balances.get(token_type, 0.0) + amount
            return True
        return False

    def get_total_value(self, prices: dict[TokenType, float]) -> float:
        """Calculate total account value.

        Args:
            prices: Current prices for each token type

        Returns:
            Total value
        """
        total = 0.0
        for tt in TokenType:
            price = prices.get(tt, 1.0)
            total += (self.balances.get(tt, 0.0) + self.staked.get(tt, 0.0)) * price
        return total

    def to_dict(self) -> dict[str, Any]:
        """Serialize account to dictionary."""
        return {
            "account_id": self.account_id,
            "balances": {k.value: v for k, v in self.balances.items()},
            "staked": {k.value: v for k, v in self.staked.items()},
            "created_at": self.created_at,
        }


class EconomicEngine:
    """Main economic engine for the QRATUM platform.

    Manages token flows, revenue streams, and economic forecasting.

    Attributes:
        engine_id: Unique engine identifier
        accounts: Dictionary of token accounts
        revenue_streams: List of revenue streams
        token_supply: Total supply for each token type
        token_prices: Current prices for each token type
        transactions: List of token flows
    """

    # Market capitalization multiplier for valuation estimates
    MARKET_CAP_MULTIPLE = 10

    def __init__(self, engine_id: str | None = None) -> None:
        """Initialize the economic engine.

        Args:
            engine_id: Optional engine ID (generated if not provided)
        """
        self.engine_id = engine_id or self._generate_id()
        self.accounts: dict[str, TokenAccount] = {}
        self.revenue_streams: list[RevenueStream] = []
        self.token_supply: dict[TokenType, float] = {
            TokenType.UTILITY: 1_000_000_000,
            TokenType.GOVERNANCE: 100_000_000,
            TokenType.STAKE: 500_000_000,
            TokenType.DATA: 500_000_000,
            TokenType.COMPLIANCE: 100_000_000,
        }
        self.token_prices: dict[TokenType, float] = {
            TokenType.UTILITY: 1.0,
            TokenType.GOVERNANCE: 10.0,
            TokenType.STAKE: 2.0,
            TokenType.DATA: 5.0,
            TokenType.COMPLIANCE: 15.0,
        }
        self.transactions: list[TokenFlow] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Initialize default revenue streams
        self._initialize_revenue_streams()

    def _generate_id(self) -> str:
        """Generate unique engine ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return f"econ_{hashlib.sha256(timestamp.encode()).hexdigest()[:12]}"

    def _initialize_revenue_streams(self) -> None:
        """Initialize default revenue streams."""
        self.revenue_streams = [
            TransactionFee(
                stream_id="txfee_001",
                revenue_type=RevenueType.TRANSACTION_FEE,
                base_rate=0.001,
            ),
            DataAsAService(
                stream_id="daas_001",
                revenue_type=RevenueType.DATA_SERVICE,
                base_rate=0.01,
            ),
            AnalyticsRevenue(
                stream_id="analytics_001",
                revenue_type=RevenueType.ANALYTICS,
                base_rate=0.01,
            ),
            ComplianceRevenue(
                stream_id="compliance_001",
                revenue_type=RevenueType.COMPLIANCE,
                base_rate=0.01,
            ),
        ]

    def create_account(self, account_id: str) -> TokenAccount:
        """Create a new token account.

        Args:
            account_id: Account identifier

        Returns:
            Created account
        """
        account = TokenAccount(account_id=account_id)
        self.accounts[account_id] = account
        return account

    def transfer(
        self,
        sender_id: str,
        receiver_id: str,
        token_type: TokenType,
        amount: float,
        network_load: float = 0.5,
    ) -> TokenFlow | None:
        """Transfer tokens between accounts.

        Args:
            sender_id: Sender account ID
            receiver_id: Receiver account ID
            token_type: Type of token
            amount: Amount to transfer
            network_load: Current network load

        Returns:
            TokenFlow if successful, None otherwise
        """
        if sender_id not in self.accounts or receiver_id not in self.accounts:
            return None

        sender = self.accounts[sender_id]
        receiver = self.accounts[receiver_id]

        # Calculate fee
        tx_fee_stream = self.revenue_streams[0]
        if isinstance(tx_fee_stream, TransactionFee):
            fee = tx_fee_stream.calculate_fee(amount, network_load)
        else:
            fee = amount * 0.001

        total_deduct = amount + fee
        if sender.withdraw(token_type, total_deduct) is None:
            return None

        receiver.deposit(token_type, amount)

        # Record fee as revenue
        tx_fee_stream.record_revenue(fee)

        # Create flow record
        flow = TokenFlow(
            flow_id=f"flow_{hashlib.sha256(f'{sender_id}{receiver_id}{amount}'.encode()).hexdigest()[:12]}",
            token_type=token_type,
            amount=amount,
            sender=sender_id,
            receiver=receiver_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            fee=fee,
        )
        self.transactions.append(flow)
        return flow

    def distribute_staking_rewards(self, reward_pool: float) -> dict[str, float]:
        """Distribute staking rewards to all stakers.

        Args:
            reward_pool: Total reward pool

        Returns:
            Dictionary of rewards by account ID
        """
        total_staked = sum(acc.staked.get(TokenType.STAKE, 0.0) for acc in self.accounts.values())

        if total_staked == 0:
            return {}

        rewards = {}
        for account_id, account in self.accounts.items():
            staked = account.staked.get(TokenType.STAKE, 0.0)
            if staked > 0:
                reward = (staked / total_staked) * reward_pool
                account.deposit(TokenType.UTILITY, reward)
                rewards[account_id] = reward

        return rewards

    def forecast_adoption(
        self,
        years: int = 5,
        initial_users: int = 1000,
        growth_rate: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Forecast adoption and economic impact.

        Args:
            years: Number of years to forecast
            initial_users: Initial user count
            growth_rate: Annual growth rate

        Returns:
            List of yearly forecasts
        """
        forecasts = []
        users = initial_users

        for year in range(1, years + 1):
            users = int(users * (1 + growth_rate))

            # Estimate transactions per user per year
            transactions_per_user = 100
            total_transactions = users * transactions_per_user

            # Estimate revenue
            avg_tx_value = 100  # Average transaction value
            tx_revenue = total_transactions * avg_tx_value * 0.001

            # Data service revenue (30% of users subscribe)
            data_revenue = users * 0.3 * 100 * 12  # Monthly subscription

            # Analytics revenue (10% use analytics)
            analytics_revenue = users * 0.1 * 50 * 12

            # Compliance revenue (5% need compliance)
            compliance_revenue = users * 0.05 * 2000

            total_revenue = tx_revenue + data_revenue + analytics_revenue + compliance_revenue

            forecasts.append(
                {
                    "year": year,
                    "users": users,
                    "transactions": total_transactions,
                    "tx_revenue": tx_revenue,
                    "data_revenue": data_revenue,
                    "analytics_revenue": analytics_revenue,
                    "compliance_revenue": compliance_revenue,
                    "total_revenue": total_revenue,
                    "market_cap_estimate": total_revenue * self.MARKET_CAP_MULTIPLE,
                }
            )

        return forecasts

    def get_statistics(self) -> dict[str, Any]:
        """Get economic statistics.

        Returns:
            Dictionary with economic statistics
        """
        total_accounts = len(self.accounts)
        total_transactions = len(self.transactions)
        total_revenue = sum(s.total_revenue for s in self.revenue_streams)

        total_value_locked = 0.0
        for account in self.accounts.values():
            total_value_locked += account.get_total_value(self.token_prices)

        revenue_by_type = {}
        for stream in self.revenue_streams:
            revenue_by_type[stream.revenue_type.value] = stream.total_revenue

        return {
            "engine_id": self.engine_id,
            "total_accounts": total_accounts,
            "total_transactions": total_transactions,
            "total_revenue": total_revenue,
            "total_value_locked": total_value_locked,
            "revenue_by_type": revenue_by_type,
            "token_supply": {k.value: v for k, v in self.token_supply.items()},
            "token_prices": {k.value: v for k, v in self.token_prices.items()},
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize engine to dictionary."""
        return {
            "engine_id": self.engine_id,
            "accounts": {k: v.to_dict() for k, v in self.accounts.items()},
            "revenue_streams": [s.to_dict() for s in self.revenue_streams],
            "token_supply": {k.value: v for k, v in self.token_supply.items()},
            "token_prices": {k.value: v for k, v in self.token_prices.items()},
            "transactions": [t.to_dict() for t in self.transactions[-100:]],  # Last 100
            "created_at": self.created_at,
        }
