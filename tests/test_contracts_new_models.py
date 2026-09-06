
import pytest
from pydantic import ValidationError

from src.core.contracts import (
    ConflictRateSummary,
    PETSummaryBasicStats,
    PETSummaryRiskSummary,
    PETUncertaintyContract,
    RiskLevelSummary,
    StatisticalTestResultContract,
)


# ---------------------- RiskLevelSummary ----------------------
def test_risk_level_summary_valid():
    obj = RiskLevelSummary(count=5, percentage=25.0)
    assert obj.count == 5
    assert obj.percentage == 25.0

def test_risk_level_summary_negative_count():
    with pytest.raises(ValidationError):
        RiskLevelSummary(count=-1, percentage=10.0)

def test_risk_level_summary_percentage_out_of_range():
    with pytest.raises(ValidationError):
        RiskLevelSummary(count=1, percentage=150.0)


# ---------------------- ConflictRateSummary ----------------------
def test_conflict_rate_summary_valid():
    obj = ConflictRateSummary(count=2, percentage=50.0, per_1000_events=500.0)
    assert obj.count == 2

def test_conflict_rate_summary_negative_per_1000():
    with pytest.raises(ValidationError):
        ConflictRateSummary(count=2, percentage=50.0, per_1000_events=-1.0)


# ---------------------- PETSummaryRiskSummary ----------------------
def test_pet_summary_risk_summary_valid():
    data = {
        "critical": {"count": 1, "percentage": 10.0},
        "serious": {"count": 2, "percentage": 20.0},
        "moderate": {"count": 3, "percentage": 30.0},
        "safe": {"count": 4, "percentage": 40.0},
        "conflict_rate": {"count": 3, "percentage": 30.0, "per_1000_events": 300.0},
    }
    obj = PETSummaryRiskSummary(**data)
    assert obj.critical.count == 1
    assert obj.conflict_rate.per_1000_events == 300.0

def test_pet_summary_risk_summary_invalid_nested():
    with pytest.raises(ValidationError):
        PETSummaryRiskSummary(
            critical={"count": -1, "percentage": 10.0},
            serious={"count": 0, "percentage": 0.0},
            moderate={"count": 0, "percentage": 0.0},
            safe={"count": 0, "percentage": 0.0},
            conflict_rate={"count": 0, "percentage": 0.0, "per_1000_events": 0.0},
        )


# ---------------------- PETSummaryBasicStats ----------------------
def test_pet_summary_basic_stats_valid():
    stats = {
        "count": 5,
        "mean": 2.0,
        "std": 0.5,
        "sem": 0.223,
        "min": 1.0,
        "q25": 1.5,
        "median": 2.0,
        "q75": 2.5,
        "max": 3.0,
        "iqr": 1.0,
        "cv": 25.0,
        "skew": 0.1,
        "kurtosis": -0.5,
        "ci_mean_lower": 1.2,
        "ci_mean_upper": 2.8,
        "ci_level": 0.95,
        "p1": 1.1,
        "p5": 1.3,
        "p10": 1.4,
        "p90": 2.6,
        "p95": 2.7,
        "p99": 2.9,
    }
    obj = PETSummaryBasicStats(**stats)
    assert obj.count == 5
    assert obj.p95 == 2.7

def test_pet_summary_basic_stats_negative_count():
    with pytest.raises(ValidationError):
        PETSummaryBasicStats(
            count=-1,
            mean=2.0,
            min=1.0,
            q25=1.5,
            median=2.0,
            q75=2.5,
            max=3.0,
            iqr=1.0,
            p1=1.0,
            p5=1.0,
            p10=1.0,
            p90=3.0,
            p95=3.0,
            p99=3.0,
        )


# ---------------------- PETUncertaintyContract ----------------------
def test_pet_uncertainty_contract_valid():
    obj = PETUncertaintyContract(
        nominal_pet=2.5,
        uncertainty_std=0.1,
        error_sources={"detection": 0.02, "homography": 0.03, "tracking": 0.01},
    )
    assert obj.nominal_pet == 2.5

def test_pet_uncertainty_contract_negative_std():
    with pytest.raises(ValidationError):
        PETUncertaintyContract(
            nominal_pet=2.5,
            uncertainty_std=-0.1,
            error_sources={"detection": 0.02},
        )

def test_pet_uncertainty_contract_empty_error_sources():
    with pytest.raises(ValidationError):
        PETUncertaintyContract(
            nominal_pet=2.5,
            uncertainty_std=0.1,
            error_sources={},
        )


# ---------------------- StatisticalTestResultContract ----------------------
def test_statistical_test_result_contract_valid():
    obj = StatisticalTestResultContract(
        test="t-test",
        alpha=0.05,
        passed=True,
        warnings=[],
        errors=[],
        type="two-sample",
        statistics={},
        test_statistics={"t_statistic": 0.0, "p_value": 1.0},
        assumptions={},
        summary="t-test result",
    )
    assert obj.test == "t-test"
    assert obj.passed is True

def test_statistical_test_result_contract_invalid_passed_no_errors():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StatisticalTestResultContract(
            test="t-test",
            alpha=0.05,
            passed=False,
            warnings=[],
            errors=[],
            type="two-sample",
            statistics={},
            test_statistics={"t_statistic": 0.0, "p_value": 1.0},
            assumptions={},
            summary="t-test result",
        )
