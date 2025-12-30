"""
Campaign模块
实现云端任务下发和车端动态触发
"""

from .campaign_manager import (
    CampaignManager,
    CampaignConfig,
    CampaignQuery,
    CampaignStatus,
    get_global_campaign_manager
)

__all__ = [
    "CampaignManager",
    "CampaignConfig",
    "CampaignQuery",
    "CampaignStatus",
    "get_global_campaign_manager"
]