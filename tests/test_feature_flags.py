"""
Tests for the feature flag system in video/pipeline.py.

This module tests the _flag helper function that controls feature gating
for audio, ASR, and prosody analysis.
"""

import os
import pytest
from video.pipeline import _flag


class TestFeatureFlags:
    """Test suite for the feature flag system."""

    def setup_method(self):
        """Clear environment variables before each test."""
        # Store original values to restore later
        self.original_env = {}
        for key in ["SPEECHUP_USE_AUDIO", "SPEECHUP_USE_ASR", "SPEECHUP_USE_PROSODY"]:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
                del os.environ[key]

    def teardown_method(self):
        """Restore original environment variables after each test."""
        for key, value in self.original_env.items():
            os.environ[key] = value

    def test_default_behavior_no_env_vars(self):
        """Test that features are ON by default when no env vars are set."""
        # Clear all relevant env vars
        for key in ["SPEECHUP_USE_AUDIO", "SPEECHUP_USE_ASR", "SPEECHUP_USE_PROSODY"]:
            if key in os.environ:
                del os.environ[key]

        # All flags should be True by default
        assert _flag("SPEECHUP_USE_AUDIO") == True
        assert _flag("SPEECHUP_USE_ASR") == True
        assert _flag("SPEECHUP_USE_PROSODY") == True

    def test_explicit_off_values(self):
        """Test that explicit OFF values disable features."""
        off_values = ["0", "false", "False", "no", "off", "OFF", "No", "FALSE"]
        
        for off_val in off_values:
            # Test each flag with each OFF value
            os.environ["SPEECHUP_USE_AUDIO"] = off_val
            assert _flag("SPEECHUP_USE_AUDIO") == False, f"Expected False for '{off_val}'"
            
            os.environ["SPEECHUP_USE_ASR"] = off_val
            assert _flag("SPEECHUP_USE_ASR") == False, f"Expected False for '{off_val}'"
            
            os.environ["SPEECHUP_USE_PROSODY"] = off_val
            assert _flag("SPEECHUP_USE_PROSODY") == False, f"Expected False for '{off_val}'"

    def test_explicit_on_values(self):
        """Test that explicit ON values enable features."""
        on_values = ["1", "true", "True", "YES", "yes", "on", "ON", "Yes", "TRUE"]
        
        for on_val in on_values:
            # Test each flag with each ON value
            os.environ["SPEECHUP_USE_AUDIO"] = on_val
            assert _flag("SPEECHUP_USE_AUDIO") == True, f"Expected True for '{on_val}'"
            
            os.environ["SPEECHUP_USE_ASR"] = on_val
            assert _flag("SPEECHUP_USE_ASR") == True, f"Expected True for '{on_val}'"
            
            os.environ["SPEECHUP_USE_PROSODY"] = on_val
            assert _flag("SPEECHUP_USE_PROSODY") == True, f"Expected True for '{on_val}'"

    def test_case_insensitive_behavior(self):
        """Test that flag values are case-insensitive."""
        # Test various case combinations
        case_variations = [
            ("TRUE", True),
            ("True", True),
            ("true", True),
            ("FALSE", False),
            ("False", False),
            ("false", False),
            ("YES", True),
            ("Yes", True),
            ("yes", True),
            ("NO", False),
            ("No", False),
            ("no", False),
            ("ON", True),
            ("On", True),
            ("on", True),
            ("OFF", False),
            ("Off", False),
            ("off", False),
        ]
        
        for value, expected in case_variations:
            os.environ["SPEECHUP_USE_AUDIO"] = value
            assert _flag("SPEECHUP_USE_AUDIO") == expected, f"Expected {expected} for '{value}'"

    def test_isolation_between_flags(self):
        """Test that changing one flag doesn't affect others."""
        # Set all flags to ON
        os.environ["SPEECHUP_USE_AUDIO"] = "1"
        os.environ["SPEECHUP_USE_ASR"] = "1"
        os.environ["SPEECHUP_USE_PROSODY"] = "1"
        
        # Verify all are ON
        assert _flag("SPEECHUP_USE_AUDIO") == True
        assert _flag("SPEECHUP_USE_ASR") == True
        assert _flag("SPEECHUP_USE_PROSODY") == True
        
        # Turn off only AUDIO
        os.environ["SPEECHUP_USE_AUDIO"] = "0"
        
        # Verify isolation: AUDIO is OFF, others remain ON
        assert _flag("SPEECHUP_USE_AUDIO") == False
        assert _flag("SPEECHUP_USE_ASR") == True
        assert _flag("SPEECHUP_USE_PROSODY") == True
        
        # Turn off only ASR
        os.environ["SPEECHUP_USE_ASR"] = "false"
        
        # Verify isolation: AUDIO and ASR are OFF, PROSODY remains ON
        assert _flag("SPEECHUP_USE_AUDIO") == False
        assert _flag("SPEECHUP_USE_ASR") == False
        assert _flag("SPEECHUP_USE_PROSODY") == True

    def test_invalid_values_default_to_off(self):
        """Test that invalid values default to OFF."""
        invalid_values = [
            "invalid", "random", "xyz", "123", "trueish", "falsy",
            "", "   ", "null", "undefined", "none", "N/A"
        ]
        
        for invalid_val in invalid_values:
            os.environ["SPEECHUP_USE_AUDIO"] = invalid_val
            assert _flag("SPEECHUP_USE_AUDIO") == False, f"Expected False for invalid value '{invalid_val}'"

    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        # Test with leading/trailing whitespace
        os.environ["SPEECHUP_USE_AUDIO"] = "  true  "
        assert _flag("SPEECHUP_USE_AUDIO") == True
        
        os.environ["SPEECHUP_USE_ASR"] = "\ttrue\t"
        assert _flag("SPEECHUP_USE_ASR") == True
        
        os.environ["SPEECHUP_USE_PROSODY"] = "\nfalse\n"
        assert _flag("SPEECHUP_USE_PROSODY") == False

    def test_none_value_handling(self):
        """Test that None values are handled correctly."""
        # Simulate None value (though os.getenv shouldn't return None)
        # This tests the edge case in the function
        if "SPEECHUP_USE_AUDIO" in os.environ:
            del os.environ["SPEECHUP_USE_AUDIO"]
        
        # Should default to True (default="1")
        assert _flag("SPEECHUP_USE_AUDIO") == True

    def test_custom_default_values(self):
        """Test that custom default values work correctly."""
        # Test with custom default "0"
        assert _flag("SPEECHUP_USE_AUDIO", "0") == False
        
        # Test with custom default "1"
        assert _flag("SPEECHUP_USE_ASR", "1") == True
        
        # Test that env var overrides custom default
        os.environ["SPEECHUP_USE_AUDIO"] = "true"
        assert _flag("SPEECHUP_USE_AUDIO", "0") == True  # env var wins
        
        os.environ["SPEECHUP_USE_ASR"] = "false"
        assert _flag("SPEECHUP_USE_ASR", "1") == False  # env var wins

    def test_all_flags_together(self):
        """Test all flags together in various combinations."""
        # Test all ON
        os.environ.update({
            "SPEECHUP_USE_AUDIO": "1",
            "SPEECHUP_USE_ASR": "true",
            "SPEECHUP_USE_PROSODY": "yes"
        })
        assert _flag("SPEECHUP_USE_AUDIO") == True
        assert _flag("SPEECHUP_USE_ASR") == True
        assert _flag("SPEECHUP_USE_PROSODY") == True
        
        # Test all OFF
        os.environ.update({
            "SPEECHUP_USE_AUDIO": "0",
            "SPEECHUP_USE_ASR": "false",
            "SPEECHUP_USE_PROSODY": "no"
        })
        assert _flag("SPEECHUP_USE_AUDIO") == False
        assert _flag("SPEECHUP_USE_ASR") == False
        assert _flag("SPEECHUP_USE_PROSODY") == False
        
        # Test mixed ON/OFF
        os.environ.update({
            "SPEECHUP_USE_AUDIO": "on",
            "SPEECHUP_USE_ASR": "off",
            "SPEECHUP_USE_PROSODY": "1"
        })
        assert _flag("SPEECHUP_USE_AUDIO") == True
        assert _flag("SPEECHUP_USE_ASR") == False
        assert _flag("SPEECHUP_USE_PROSODY") == True


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])
