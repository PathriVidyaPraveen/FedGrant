"""Trace preparation and deterministic per-client trace assignment."""

from fedgrant.traces.binning import bin_opportunities, split_rates
from fedgrant.traces.client_trace import client_trace

__all__ = ["bin_opportunities", "split_rates", "client_trace"]
