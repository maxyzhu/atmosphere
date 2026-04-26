"""
Tests for the stage abstraction.

These test the architectural contract (stages layer correctly, registry
returns the right order) without hitting real OSM / Mapillary APIs —
the fetch functions they delegate to are already tested elsewhere.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from atmosphere.stages import (
    STAGE_ORDER,
    STAGE_REGISTRY,
    OSMStage,
    MapillaryStage,
    Stage,
    StageData,
    get_stages_up_to,
    list_stages,
)


class TestRegistry:
    def test_all_registered_stages_in_order(self):
        """Every stage in STAGE_ORDER must exist in STAGE_REGISTRY."""
        for name in STAGE_ORDER:
            assert name in STAGE_REGISTRY
            assert isinstance(STAGE_REGISTRY[name], Stage)

    def test_every_stage_has_name_and_description(self):
        for name, stage in STAGE_REGISTRY.items():
            assert stage.name == name
            assert stage.description
            assert len(stage.description) > 10  # non-trivial

    def test_osm_is_first(self):
        """OSM must be the foundational layer — no stage depends on it
        not being first. Breaking this would break every later stage."""
        assert STAGE_ORDER[0] == "osm"


class TestGetStagesUpTo:
    def test_single_stage(self):
        result = get_stages_up_to("osm")
        assert len(result) == 1
        assert result[0].name == "osm"

    def test_includes_all_earlier(self):
        result = get_stages_up_to("mapillary")
        names = [s.name for s in result]
        # Must include osm as well
        assert "osm" in names
        assert "mapillary" in names
        # In the right order
        assert names.index("osm") < names.index("mapillary")

    def test_unknown_stage_raises(self):
        with pytest.raises(KeyError, match="Unknown stage"):
            get_stages_up_to("nonexistent")


class TestListStages:
    def test_returns_pairs(self):
        result = list_stages()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, desc = item
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_order_matches_registry(self):
        result = list_stages()
        names = [r[0] for r in result]
        assert names == STAGE_ORDER


class TestStageData:
    def test_empty_default(self):
        data = StageData()
        assert data.buildings == []
        assert data.mapillary_images == []

    def test_mutable_fields_independent_between_instances(self):
        """Regression test: default_factory[list] must not share state."""
        d1 = StageData()
        d2 = StageData()
        d1.buildings.append("x")  # type: ignore[arg-type]
        assert d2.buildings == []


class TestOSMStageIntegration:
    """OSMStage.fetch delegates to fetch_buildings; verify the contract."""

    def test_populates_buildings(self, tmp_path):
        with patch(
            "atmosphere.stages.fetch_buildings",
            return_value=["fake_building_1", "fake_building_2"],
        ) as mocked:
            data = StageData()
            OSMStage().fetch(
                lat=47.6097, lon=-122.3331, radius_m=100,
                data=data, use_cache=False,
            )
            mocked.assert_called_once()
            assert data.buildings == ["fake_building_1", "fake_building_2"]

    def test_passes_options_to_fetch(self):
        with patch(
            "atmosphere.stages.fetch_buildings",
            return_value=[],
        ) as mocked:
            OSMStage().fetch(
                lat=47.6097, lon=-122.3331, radius_m=200,
                data=StageData(),
                use_cache=False,
            )
            call_kwargs = mocked.call_args.kwargs
            assert call_kwargs["lat"] == 47.6097
            assert call_kwargs["lon"] == -122.3331
            assert call_kwargs["radius_m"] == 200
            assert call_kwargs["use_cache"] is False


class TestMapillaryStageIntegration:
    def test_populates_images(self):
        with patch(
            "atmosphere.stages.fetch_mapillary_images",
            return_value=["img1", "img2"],
        ) as mocked:
            data = StageData()
            MapillaryStage().fetch(
                lat=47.6097, lon=-122.3331, radius_m=100,
                data=data,
                mapillary_limit=50,
                download_thumbnails=False,
                use_cache=True,
            )
            mocked.assert_called_once()
            assert data.mapillary_images == ["img1", "img2"]

    def test_preserves_earlier_stage_data(self):
        """Running MapillaryStage must not clobber fields populated earlier."""
        data = StageData()
        data.buildings = ["existing_building"]  # type: ignore[list-item]

        with patch(
            "atmosphere.stages.fetch_mapillary_images",
            return_value=[],
        ):
            MapillaryStage().fetch(
                lat=47.6097, lon=-122.3331, radius_m=100,
                data=data,
                mapillary_limit=50,
                download_thumbnails=False,
                use_cache=True,
            )

        # Buildings must still be there
        assert data.buildings == ["existing_building"]