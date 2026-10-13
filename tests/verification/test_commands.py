from dataclasses import fields

import pytest

from ware_ops_algos.planning.commands import (
    CancelTour, CommitTour, CreatePickList, CreateTour, ReplaceSuffix,
    Reschedule,
)


def test_command_contract_has_no_snapshot_version_fields():
    for command in (CancelTour, CommitTour, CreatePickList, CreateTour,
                    ReplaceSuffix, Reschedule):
        assert not any(field.name.startswith("expected_")
                       for field in fields(command))
    assert CommitTour(tour_id=3).tour_id == 3
    assert CommitTour(creation_index=0).creation_index == 0
    with pytest.raises(ValueError, match="exactly one"):
        CommitTour()
    with pytest.raises(ValueError, match="exactly one"):
        CommitTour(tour_id=3, creation_index=0)
