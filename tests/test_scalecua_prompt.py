"""The ScaleCUA system prompt hands the model no credential (the upstream line with the image's sudo password is removed)."""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), "..", "src", "drums_verifiers", "scalecua.py")
_spec = importlib.util.spec_from_file_location("scalecua_only", _p)
scalecua = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scalecua)


def test_the_system_prompt_names_no_password():
    p = scalecua.build_system_prompt()
    assert "password" not in p.lower()
    assert "sudo" not in p.lower()
    assert "Home directory of this Ubuntu system is '/home/user'." in p     # the rest of the upstream text is still there
