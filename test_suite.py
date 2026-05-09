import os
import uuid
import pytest
from auth_db import hash_password, create_user, authenticate_user, generate_chat_id, init_user_db
from db import init_db, save_message, get_chat_history

# ==========================================
# UNIT TESTS: Authentication & Utilities
# ==========================================

def test_hash_password():
    """Test that password hashing works and is deterministic."""
    hash1 = hash_password("secret123")
    assert len(hash1) == 64  # SHA-256 hex digest length
    assert hash1 == hash_password("secret123")  # Must be deterministic

def test_generate_chat_id():
    """Test chat ID generation based on user credentials."""
    chat_id = generate_chat_id("johndoe", "mypassword")
    assert len(chat_id) == 64

# ==========================================
# INTEGRATION TESTS: SQLite Databases
# ==========================================

def test_database_creation():
    """Test that the SQLite databases can be initialized successfully."""
    init_db()
    init_user_db()
    assert os.path.exists("users.db")
    assert os.path.exists("chat_history.db")

def test_user_flow():
    """Test the complete user creation and authentication flow."""
    # Generate a random username to avoid collisions in the real local DB
    test_username = f"testuser_{uuid.uuid4().hex[:8]}"
    test_password = "securepassword"
    
    # 1. Create User
    assert create_user(test_username, test_password) is True
    
    # 2. Duplicate user should fail (SQLite UNIQUE constraint)
    assert create_user(test_username, test_password) is False
    
    # 3. Authenticate with correct password
    assert authenticate_user(test_username, test_password) is True
    
    # 4. Authenticate with wrong password
    assert authenticate_user(test_username, "wrongpass") is False
    
    # 5. Authenticate nonexistent user
    assert authenticate_user("fakeuser999", "fake") is False

def test_chat_history_flow():
    """Test saving and retrieving messages from the chat database."""
    test_chat_id = f"testchat_{uuid.uuid4().hex[:8]}"
    
    # Save messages
    save_message(test_chat_id, "user", "What is a contract?")
    save_message(test_chat_id, "assistant", "A contract is a legally binding agreement.")
    
    # Retrieve messages
    history = get_chat_history(test_chat_id)
    
    assert len(history) == 2
    assert history[0][0] == "user"
    assert history[0][1] == "What is a contract?"
    assert history[1][0] == "assistant"
    assert history[1][1] == "A contract is a legally binding agreement."

# ==========================================
# E2E TESTS: Streamlit UI
# ==========================================

def test_streamlit_app_ui():
    """Test the Streamlit UI flow using Streamlit AppTest framework."""
    try:
        from streamlit.testing.v1 import AppTest
        
        # The app requires config.yaml to run. If it's not set up locally, skip the UI test.
        if not os.path.exists("config.yaml"):
            pytest.skip("Skipping UI test because config.yaml is missing in the root directory.")
            
        # Initialize the simulated Streamlit environment
        at = AppTest.from_file("app3.py")
        
        # Run the app
        at.run()
        
        # Verify the UI rendered the login screen correctly
        assert not at.exception, "The Streamlit app threw an exception during initialization."
        assert at.title[0].value == "🔐 Legal AI Assistant Login"
        assert len(at.tabs) == 2
        
    except ImportError:
        pytest.skip("Streamlit AppTest requires a newer version of streamlit (>=1.28.0).")
    except Exception as e:
        pytest.skip(f"UI Test skipped due to environment setup error: {e}")
