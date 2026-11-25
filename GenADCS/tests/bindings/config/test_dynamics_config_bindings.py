"""Cross-backend tests for Python, pybind11, and PyO3 config bindings."""

import gc
import weakref

import pytest


def test_default_lists_are_empty_and_instance_independent(config_backend):
    first = config_backend.DynamicsConfig()
    second = config_backend.DynamicsConfig()
    assert first.forces == []
    assert first.torques == []
    assert first.forces is not second.forces
    assert first.torques is not second.torques


def test_constructor_accepts_positional_and_keyword_lists(config_backend):
    positional = config_backend.DynamicsConfig(["force"], ["torque"])
    keyword = config_backend.DynamicsConfig(torques=["torque"], forces=["force"])
    assert positional.forces == keyword.forces == ["force"]
    assert positional.torques == keyword.torques == ["torque"]


def test_list_getter_ownership_and_mutation_quirks(config_backend, config_backend_name):
    force = object()
    source = [force]
    config = config_backend.DynamicsConfig(forces=source)

    if config_backend_name == "python":
        assert config.forces is source
        config.forces.append("appended")
        assert len(config.forces) == 2
    else:
        # Both native bindings store a vector and materialize a fresh Python list per getter.
        assert config.forces is not source
        assert config.forces[0] is force
        exposed = config.forces
        exposed.append("does not update native storage")
        assert config.forces == [force]

    replacement = ["replacement"]
    config.forces = replacement
    assert config.forces == replacement
    if config_backend_name == "python":
        assert config.forces is replacement
    else:
        assert config.forces is not replacement


def test_collection_setter_sequence_conversion(config_backend, config_backend_name):
    config = config_backend.DynamicsConfig()
    config.forces = ("a", "b")
    assert (
        config.forces == ["a", "b"]
        if config_backend_name != "python"
        else config.forces == ("a", "b")
    )


def test_none_constructor_argument_is_backend_specific(
    config_backend, config_backend_name
):
    if config_backend_name == "cpp":
        with pytest.raises(TypeError):
            config_backend.DynamicsConfig(forces=None)
    elif config_backend_name == "python":
        assert config_backend.DynamicsConfig(forces=None).forces is None
    else:
        assert config_backend.DynamicsConfig(forces=None).forces == []


def test_item_references_survive_collection_and_are_released_on_destruction(
    config_backend,
):
    class RetainedItem:
        pass

    item = RetainedItem()
    item_ref = weakref.ref(item)
    source = [item]
    config = config_backend.DynamicsConfig(forces=source)
    del item, source
    gc.collect()
    assert item_ref() is not None

    # A list returned by a getter owns its own reference to each contained item,
    # even after the native/Python config object has been destroyed.
    exposed = config.forces
    del config
    gc.collect()
    assert exposed[0] is item_ref()
    del exposed
    gc.collect()
    assert item_ref() is None


def test_replacing_items_releases_old_reference_but_keeps_new_reference(config_backend):
    class RetainedItem:
        pass

    old_item, new_item = RetainedItem(), RetainedItem()
    old_ref, new_ref = weakref.ref(old_item), weakref.ref(new_item)
    config = config_backend.DynamicsConfig(forces=[old_item])
    config.forces = [new_item]
    del old_item, new_item
    gc.collect()

    assert old_ref() is None
    assert config.forces[0] is new_ref()
