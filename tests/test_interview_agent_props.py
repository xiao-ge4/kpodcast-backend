"""
Property-based tests for Interview Podcast Mode feature.
Uses Hypothesis library for property testing.

**Feature: interview-podcast-mode**
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock external dependencies before importing interview_agent
# This prevents import errors when optional dependencies are not installed
mock_fetch_url_enhanced = MagicMock(return_value={
    "success": True,
    "text": "Mocked content",
    "status": 200
})

mock_bocha_client = MagicMock()
mock_bocha_client.search.return_value = []

# Create mock modules for optional dependencies that may not be installed
for mod_name in ['trafilatura', 'PyPDF2', 'pdfplumber', 'newspaper', 'readability']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from pipeline.interview_agent import (
    InterviewAgent,
    InterviewSession,
    detect_urls,
    start_session,
    get_session,
    clear_sessions,
)


class TestSessionCreation:
    """
    **Feature: interview-podcast-mode, Property 1: Session creation returns valid ID**
    **Validates: Requirements 7.1**
    
    For any call to start_interview, the returned session_id shall be a non-empty 
    string that can be used in subsequent API calls.
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        num_sessions=st.integers(min_value=1, max_value=20)
    )
    def test_session_creation_returns_valid_id(self, num_sessions):
        """
        **Feature: interview-podcast-mode, Property 1: Session creation returns valid ID**
        **Validates: Requirements 7.1**
        
        Property: Every call to start_session returns a session with:
        - A non-empty session_id string
        - The session_id can be used to retrieve the session
        - Each session_id is unique
        """
        clear_sessions()
        
        # Create mock config to avoid loading real config
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session_ids = set()
        
        for _ in range(num_sessions):
            session = agent.start_session()
            
            # Property 1a: session_id is non-empty string
            assert isinstance(session.session_id, str), "session_id must be a string"
            assert len(session.session_id) > 0, "session_id must be non-empty"
            
            # Property 1b: session_id is unique
            assert session.session_id not in session_ids, f"Duplicate session_id: {session.session_id}"
            session_ids.add(session.session_id)
            
            # Property 1c: session can be retrieved using session_id
            retrieved = agent.get_session(session.session_id)
            assert retrieved is not None, "Session should be retrievable"
            assert retrieved.session_id == session.session_id, "Retrieved session should have same ID"
            
            # Property 1d: session has valid initial state
            assert isinstance(session.messages, list), "messages should be a list"
            assert len(session.messages) == 0, "new session should have no messages"
            assert isinstance(session.key_points, list), "key_points should be a list"
            assert isinstance(session.materials, list), "materials should be a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestConversationHistory:
    """
    **Feature: interview-podcast-mode, Property 2: Conversation history grows with each message**
    **Validates: Requirements 7.3**
    
    For any session with N messages, after sending a new message, the session 
    shall contain N+2 messages (user message + AI reply).
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        messages=st.lists(
            st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            min_size=1,
            max_size=10
        )
    )
    def test_conversation_history_grows_with_each_message(self, messages):
        """
        **Feature: interview-podcast-mode, Property 2: Conversation history grows with each message**
        **Validates: Requirements 7.3**
        
        Property: For each message sent, the conversation history grows by exactly 2
        (one user message + one assistant reply).
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        # Mock the LLM client to avoid real API calls
        def mock_chat(llm_messages, stream=False):
            return {
                "Choices": [{"Message": {"Content": "这是AI的回复。"}}]
            }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            expected_count = 0
            
            for msg in messages:
                initial_count = len(session.messages)
                assert initial_count == expected_count, f"Expected {expected_count} messages, got {initial_count}"
                
                result = agent.chat(session.session_id, msg)
                
                # Property: After sending a message, history grows by exactly 2
                new_count = len(session.messages)
                assert new_count == initial_count + 2, \
                    f"Expected {initial_count + 2} messages after chat, got {new_count}"
                
                # Property: Last two messages are user and assistant
                assert session.messages[-2]["role"] == "user", "Second to last should be user message"
                assert session.messages[-1]["role"] == "assistant", "Last should be assistant message"
                
                # Property: User message content matches what was sent
                assert session.messages[-2]["content"] == msg, "User message content should match"
                
                # Property: message_count in result matches actual count
                assert result["message_count"] == new_count, "message_count should match actual count"
                
                expected_count = new_count



class TestURLDetection:
    """
    **Feature: interview-podcast-mode, Property 4: URL detection in messages**
    **Validates: Requirements 3.1**
    
    For any user message containing a valid URL pattern, the system shall 
    detect and flag it for processing.
    """
    
    @settings(max_examples=100)
    @given(
        protocol=st.sampled_from(['http://', 'https://']),
        domain=st.text(
            alphabet=st.characters(whitelist_categories=('Ll',), whitelist_characters='-'),
            min_size=3,
            max_size=20
        ).filter(lambda x: x and not x.startswith('-') and not x.endswith('-')),
        tld=st.sampled_from(['.com', '.org', '.net', '.io', '.cn', '.edu']),
        path=st.text(
            alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='/-_'),
            min_size=0,
            max_size=30
        )
    )
    def test_url_detection_finds_valid_urls(self, protocol, domain, tld, path):
        """
        **Feature: interview-podcast-mode, Property 4: URL detection in messages**
        **Validates: Requirements 3.1**
        
        Property: For any valid URL, detect_urls should find it in the message.
        """
        # Construct a valid URL
        url = f"{protocol}{domain}{tld}{path}"
        
        # Test URL alone
        detected = detect_urls(url)
        assert len(detected) >= 1, f"Should detect URL: {url}"
        assert any(url.startswith(d) or d.startswith(url.split('?')[0]) for d in detected), \
            f"Detected URLs {detected} should include {url}"
    
    @settings(max_examples=100)
    @given(
        prefix=st.text(min_size=0, max_size=50),
        protocol=st.sampled_from(['http://', 'https://']),
        domain=st.sampled_from(['example.com', 'test.org', 'github.com', 'google.cn']),
        suffix=st.text(min_size=0, max_size=50)
    )
    def test_url_detection_in_mixed_text(self, prefix, protocol, domain, suffix):
        """
        **Feature: interview-podcast-mode, Property 4: URL detection in messages**
        **Validates: Requirements 3.1**
        
        Property: URLs embedded in text should still be detected.
        """
        url = f"{protocol}{domain}"
        message = f"{prefix} {url} {suffix}"
        
        detected = detect_urls(message)
        assert len(detected) >= 1, f"Should detect URL in message: {message}"
        assert any(domain in d for d in detected), \
            f"Detected URLs {detected} should include URL with domain {domain}"
    
    @settings(max_examples=100)
    @given(
        text=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
            min_size=0,
            max_size=200
        ).filter(lambda x: 'http' not in x.lower() and 'www.' not in x.lower())
    )
    def test_url_detection_no_false_positives(self, text):
        """
        **Feature: interview-podcast-mode, Property 4: URL detection in messages**
        **Validates: Requirements 3.1**
        
        Property: Text without URLs should return empty list.
        """
        detected = detect_urls(text)
        assert detected == [], f"Should not detect URLs in text without URLs: {text}, got: {detected}"
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=50)
    @given(
        url=st.sampled_from([
            'https://example.com/article',
            'http://test.org/page',
            'https://github.com/user/repo',
        ]),
        surrounding_text=st.text(min_size=10, max_size=100).filter(
            lambda x: 'http' not in x.lower() and 'www.' not in x.lower()
        )
    )
    def test_chat_detects_urls_in_messages(self, url, surrounding_text):
        """
        **Feature: interview-podcast-mode, Property 4: URL detection in messages**
        **Validates: Requirements 3.1**
        
        Property: When a user sends a message containing a URL, the chat method
        should detect and return it in the response.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        def mock_chat(llm_messages, stream=False):
            return {
                "Choices": [{"Message": {"Content": "我看到你分享了一个链接。"}}]
            }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            message = f"{surrounding_text} {url}"
            result = agent.chat(session.session_id, message)
            
            # Property: detected_urls should contain the URL
            assert "detected_urls" in result, "Result should have detected_urls field"
            assert len(result["detected_urls"]) >= 1, f"Should detect URL in message: {message}"
            assert any(url in detected or detected in url for detected in result["detected_urls"]), \
                f"detected_urls {result['detected_urls']} should include {url}"
            
            # Property: The URL should also be stored in the message history
            user_msg = session.messages[-2]  # Second to last is user message
            assert "detected_urls" in user_msg, "User message should have detected_urls"
            assert len(user_msg["detected_urls"]) >= 1, "User message should have detected URLs"


class TestMaterialStorage:
    """
    **Feature: interview-podcast-mode, Property 3: Materials are stored and retrievable**
    **Validates: Requirements 7.4, 3.4**
    
    For any material added to a session, the material shall appear in the 
    session's materials list with a valid summary.
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        material_type=st.sampled_from(['url', 'document', 'topic']),
        content=st.text(min_size=10, max_size=500).filter(lambda x: x.strip())
    )
    def test_material_storage_and_retrieval(self, material_type, content):
        """
        **Feature: interview-podcast-mode, Property 3: Materials are stored and retrievable**
        **Validates: Requirements 7.4, 3.4**
        
        Property: For any material added to a session:
        - The material appears in the session's materials list
        - The material has a valid (non-empty) summary
        - The material has a valid ID
        - The material can be retrieved via the session
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
            "bocha_base_url": "https://api.bocha.test",
            "bocha_api_id": "test_id",
            "bocha_api_key": "test_key",
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        initial_material_count = len(session.materials)
        
        # Mock external dependencies
        with patch.object(agent, '_get_llm_client') as mock_get_client, \
             patch('pipeline.interview_agent.fetch_url_enhanced') as mock_fetch, \
             patch('pipeline.interview_agent.BochaClient') as mock_bocha_class:
            
            # Setup LLM mock
            mock_client = MagicMock()
            mock_client.chat.return_value = {
                "Choices": [{"Message": {"Content": "这是一个测试摘要。"}}]
            }
            mock_get_client.return_value = mock_client
            
            # Setup URL fetch mock
            mock_fetch.return_value = {
                "success": True,
                "text": f"Fetched content for: {content}",
                "status": 200
            }
            
            # Setup Bocha search mock
            mock_bocha_instance = MagicMock()
            mock_bocha_instance.search.return_value = [
                {"title": "Test Result", "snippet": "Test snippet", "url": "https://test.com"}
            ]
            mock_bocha_class.return_value = mock_bocha_instance
            
            # Add material
            result = agent.add_material(session.session_id, material_type, content)
            
            # Property 3a: Result has required fields
            assert "id" in result, "Result should have 'id' field"
            assert "summary" in result, "Result should have 'summary' field"
            assert "source" in result, "Result should have 'source' field"
            assert "ai_thoughts" in result, "Result should have 'ai_thoughts' field"
            
            # Property 3b: ID is non-empty
            assert result["id"], "Material ID should be non-empty"
            assert len(result["id"]) > 0, "Material ID should have length > 0"
            
            # Property 3c: Summary is non-empty
            assert result["summary"], "Summary should be non-empty"
            assert len(result["summary"]) > 0, "Summary should have length > 0"
            
            # Property 3d: Material count increased by 1
            assert len(session.materials) == initial_material_count + 1, \
                f"Materials count should increase by 1, got {len(session.materials)}"
            
            # Property 3e: Material is retrievable from session
            added_material = session.materials[-1]
            assert added_material["id"] == result["id"], "Material ID should match"
            assert added_material["type"] == material_type, "Material type should match"
            assert added_material["summary"] == result["summary"], "Summary should match"
    
    @settings(max_examples=50)
    @given(
        num_materials=st.integers(min_value=1, max_value=10),
        material_types=st.lists(
            st.sampled_from(['url', 'document', 'topic']),
            min_size=1,
            max_size=10
        )
    )
    def test_multiple_materials_all_stored(self, num_materials, material_types):
        """
        **Feature: interview-podcast-mode, Property 3: Materials are stored and retrievable**
        **Validates: Requirements 7.4, 3.4**
        
        Property: When multiple materials are added, all are stored and retrievable.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
            "bocha_base_url": "https://api.bocha.test",
            "bocha_api_id": "test_id",
            "bocha_api_key": "test_key",
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        # Mock external dependencies
        with patch.object(agent, '_get_llm_client') as mock_get_client, \
             patch('pipeline.interview_agent.fetch_url_enhanced') as mock_fetch, \
             patch('pipeline.interview_agent.BochaClient') as mock_bocha_class:
            
            mock_client = MagicMock()
            mock_client.chat.return_value = {
                "Choices": [{"Message": {"Content": "测试摘要"}}]
            }
            mock_get_client.return_value = mock_client
            
            mock_fetch.return_value = {
                "success": True,
                "text": "Fetched content",
                "status": 200
            }
            
            mock_bocha_instance = MagicMock()
            mock_bocha_instance.search.return_value = [
                {"title": "Result", "snippet": "Snippet", "url": "https://test.com"}
            ]
            mock_bocha_class.return_value = mock_bocha_instance
            
            # Add multiple materials
            added_ids = []
            types_to_add = material_types[:num_materials]
            
            for i, mat_type in enumerate(types_to_add):
                content = f"Test content {i} for {mat_type}"
                result = agent.add_material(session.session_id, mat_type, content)
                added_ids.append(result["id"])
            
            # Property: All materials are stored
            assert len(session.materials) == len(types_to_add), \
                f"Expected {len(types_to_add)} materials, got {len(session.materials)}"
            
            # Property: All IDs are unique
            stored_ids = [m["id"] for m in session.materials]
            assert len(stored_ids) == len(set(stored_ids)), "All material IDs should be unique"
            
            # Property: All added IDs are in stored materials
            for aid in added_ids:
                assert aid in stored_ids, f"Added ID {aid} should be in stored materials"
    
    @settings(max_examples=50)
    @given(
        content=st.text(min_size=5, max_size=200).filter(lambda x: x.strip())
    )
    def test_material_type_url_stores_source(self, content):
        """
        **Feature: interview-podcast-mode, Property 3: Materials are stored and retrievable**
        **Validates: Requirements 7.4, 3.4**
        
        Property: URL materials store the URL as the source.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        url = f"https://example.com/{content.replace(' ', '-')[:50]}"
        
        with patch.object(agent, '_get_llm_client') as mock_get_client, \
             patch('pipeline.interview_agent.fetch_url_enhanced') as mock_fetch:
            
            mock_client = MagicMock()
            mock_client.chat.return_value = {
                "Choices": [{"Message": {"Content": "URL摘要"}}]
            }
            mock_get_client.return_value = mock_client
            
            mock_fetch.return_value = {
                "success": True,
                "text": "Page content",
                "status": 200
            }
            
            result = agent.add_material(session.session_id, "url", url)
            
            # Property: Source contains the URL
            assert url in result["source"] or result["source"] == url, \
                f"Source should contain URL, got: {result['source']}"
            
            # Property: Material in session has correct source
            material = session.materials[-1]
            assert material["source"] == result["source"], "Stored source should match returned source"


class TestStyleAnalysis:
    """
    **Feature: interview-podcast-mode, Property 9: Style analysis produces result**
    **Validates: Requirements 5.1**
    
    For any session with 5+ user messages, the style analysis shall return 
    a non-empty tone and vocabulary list.
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        messages=st.lists(
            st.text(min_size=20, max_size=300).filter(lambda x: x.strip()),
            min_size=5,
            max_size=15
        )
    )
    def test_style_analysis_produces_result_with_sufficient_messages(self, messages):
        """
        **Feature: interview-podcast-mode, Property 9: Style analysis produces result**
        **Validates: Requirements 5.1**
        
        Property: For any session with 5+ user messages, analyze_style returns:
        - A non-empty tone string
        - A vocabulary list (may be empty but must exist)
        - An expressions list (may be empty but must exist)
        - A sentence_style string
        """
        from unittest.mock import patch, MagicMock
        import json
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        # Mock LLM responses
        chat_call_count = [0]
        
        def mock_chat(llm_messages, stream=False):
            chat_call_count[0] += 1
            # Check if this is a style analysis call (contains style analysis prompt)
            last_msg = llm_messages[-1].get("Content", "") if llm_messages else ""
            if "分析" in last_msg and "风格" in last_msg:
                # Return style analysis JSON
                return {
                    "Choices": [{
                        "Message": {
                            "Content": json.dumps({
                                "tone": "轻松幽默",
                                "vocabulary": ["有趣", "确实", "感觉"],
                                "expressions": ["我觉得", "其实"],
                                "sentence_style": "口语化，喜欢用短句"
                            }, ensure_ascii=False)
                        }
                    }]
                }
            # Regular chat response
            return {
                "Choices": [{"Message": {"Content": "这是AI的回复。请继续分享你的想法。"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            # Add messages to session
            for msg in messages:
                agent.chat(session.session_id, msg)
            
            # Verify we have enough user messages
            user_msg_count = len([m for m in session.messages if m["role"] == "user"])
            assert user_msg_count >= 5, f"Should have at least 5 user messages, got {user_msg_count}"
            
            # Analyze style
            style = agent.analyze_style(session.session_id)
            
            # Property 9a: tone is non-empty string
            assert "tone" in style, "Style should have 'tone' field"
            assert isinstance(style["tone"], str), "tone should be a string"
            assert len(style["tone"]) > 0, "tone should be non-empty"
            
            # Property 9b: vocabulary is a list
            assert "vocabulary" in style, "Style should have 'vocabulary' field"
            assert isinstance(style["vocabulary"], list), "vocabulary should be a list"
            
            # Property 9c: expressions is a list
            assert "expressions" in style, "Style should have 'expressions' field"
            assert isinstance(style["expressions"], list), "expressions should be a list"
            
            # Property 9d: sentence_style exists
            assert "sentence_style" in style, "Style should have 'sentence_style' field"
            assert isinstance(style["sentence_style"], str), "sentence_style should be a string"
            
            # Property 9e: style is stored in session
            assert session.user_style == style, "Style should be stored in session"
    
    @settings(max_examples=50)
    @given(
        messages=st.lists(
            st.text(min_size=10, max_size=200).filter(lambda x: x.strip()),
            min_size=1,
            max_size=4
        )
    )
    def test_style_analysis_with_insufficient_messages(self, messages):
        """
        **Feature: interview-podcast-mode, Property 9: Style analysis produces result**
        **Validates: Requirements 5.1**
        
        Property: For sessions with fewer than 5 user messages, analyze_style
        returns a valid structure but indicates insufficient data.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        def mock_chat(llm_messages, stream=False):
            return {
                "Choices": [{"Message": {"Content": "AI回复"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            # Add fewer than 5 messages
            for msg in messages[:4]:
                agent.chat(session.session_id, msg)
            
            # Analyze style
            style = agent.analyze_style(session.session_id)
            
            # Property: Should still return valid structure
            assert "tone" in style, "Style should have 'tone' field"
            assert "vocabulary" in style, "Style should have 'vocabulary' field"
            assert "expressions" in style, "Style should have 'expressions' field"
            assert "sentence_style" in style, "Style should have 'sentence_style' field"


class TestKeyPointsExtraction:
    """
    **Feature: interview-podcast-mode, Property 5: Key points extraction grows with conversation**
    **Validates: Requirements 2.4**
    
    For any session with user opinions expressed, the key_points list shall 
    contain at least one entry after 3+ exchanges.
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        # Generate messages that are long enough to be considered opinions (>50 chars)
        messages=st.lists(
            st.text(min_size=60, max_size=300).filter(lambda x: x.strip() and len(x.strip()) > 50),
            min_size=3,
            max_size=10
        )
    )
    def test_key_points_grow_with_conversation(self, messages):
        """
        **Feature: interview-podcast-mode, Property 5: Key points extraction grows with conversation**
        **Validates: Requirements 2.4**
        
        Property: For any session with 3+ user messages containing opinions 
        (messages > 50 chars), the key_points list should contain at least one entry.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        def mock_chat(llm_messages, stream=False):
            return {
                "Choices": [{"Message": {"Content": "这是AI的回复。"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            # Send messages
            for msg in messages:
                agent.chat(session.session_id, msg)
            
            # Property: After 3+ exchanges with substantial messages, 
            # key_points should have at least one entry
            assert len(session.key_points) >= 1, \
                f"Expected at least 1 key point after {len(messages)} messages, got {len(session.key_points)}"
            
            # Property: Each key point has required fields
            for kp in session.key_points:
                assert "point" in kp, "Key point should have 'point' field"
                assert "source_message_idx" in kp, "Key point should have 'source_message_idx' field"
                assert "confidence" in kp, "Key point should have 'confidence' field"
                assert len(kp["point"]) > 0, "Key point text should be non-empty"


class TestShortConversationWarning:
    """
    **Feature: interview-podcast-mode, Property 6: Short conversation warning**
    **Validates: Requirements 4.3**
    
    For any session with fewer than 3 message exchanges, the generate endpoint 
    shall return a warning message.
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        num_messages=st.integers(min_value=0, max_value=2)
    )
    def test_short_conversation_returns_warning(self, num_messages):
        """
        **Feature: interview-podcast-mode, Property 6: Short conversation warning**
        **Validates: Requirements 4.3**
        
        Property: For any session with fewer than 3 user messages, 
        generate_script returns a warning.
        """
        from unittest.mock import patch, MagicMock
        import json
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        def mock_chat(llm_messages, stream=False):
            last_msg = llm_messages[-1].get("Content", "") if llm_messages else ""
            if "风格" in last_msg:
                return {
                    "Choices": [{
                        "Message": {
                            "Content": json.dumps({
                                "tone": "自然",
                                "vocabulary": [],
                                "expressions": [],
                                "sentence_style": "口语化"
                            }, ensure_ascii=False)
                        }
                    }]
                }
            return {
                "Choices": [{"Message": {"Content": "这是生成的脚本内容。"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            # Add fewer than 3 messages
            for i in range(num_messages):
                agent.chat(session.session_id, f"测试消息 {i}")
            
            # Generate script
            result = agent.generate_script(session.session_id)
            
            # Property: Warning should be present for short conversations
            assert "warning" in result, \
                f"Expected warning for {num_messages} messages, but no warning returned"
            assert result["warning"] is not None, "Warning should not be None"
            assert len(result["warning"]) > 0, "Warning should be non-empty"
    
    @settings(max_examples=50)
    @given(
        num_messages=st.integers(min_value=3, max_value=10)
    )
    def test_sufficient_conversation_no_warning(self, num_messages):
        """
        **Feature: interview-podcast-mode, Property 6: Short conversation warning**
        **Validates: Requirements 4.3**
        
        Property: For any session with 3+ user messages, 
        generate_script should not return a warning (or warning is None).
        """
        from unittest.mock import patch, MagicMock
        import json
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        def mock_chat(llm_messages, stream=False):
            last_msg = llm_messages[-1].get("Content", "") if llm_messages else ""
            if "风格" in last_msg:
                return {
                    "Choices": [{
                        "Message": {
                            "Content": json.dumps({
                                "tone": "自然",
                                "vocabulary": [],
                                "expressions": [],
                                "sentence_style": "口语化"
                            }, ensure_ascii=False)
                        }
                    }]
                }
            return {
                "Choices": [{"Message": {"Content": "这是生成的脚本内容。"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            # Add 3+ messages
            for i in range(num_messages):
                agent.chat(session.session_id, f"测试消息 {i}")
            
            # Generate script
            result = agent.generate_script(session.session_id)
            
            # Property: No warning for sufficient conversations
            assert "warning" not in result or result.get("warning") is None, \
                f"Expected no warning for {num_messages} messages, but got: {result.get('warning')}"


class TestScriptContainsUserContent:
    """
    **Feature: interview-podcast-mode, Property 7: Generated script contains user content**
    **Validates: Requirements 4.2, 5.3**
    
    For any session with key_points, the generated script shall contain 
    references to at least one user opinion.
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        # Generate distinctive user opinions that should appear in the script
        opinions=st.lists(
            st.text(min_size=60, max_size=200).filter(lambda x: x.strip() and len(x.strip()) > 50),
            min_size=3,
            max_size=8
        )
    )
    def test_script_contains_user_opinions(self, opinions):
        """
        **Feature: interview-podcast-mode, Property 7: Generated script contains user content**
        **Validates: Requirements 4.2, 5.3**
        
        Property: For any session with key_points, the generated script 
        should reference user opinions (key points are passed to LLM prompt).
        """
        from unittest.mock import patch, MagicMock
        import json
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        # Track what's passed to the script generation prompt
        script_prompt_content = []
        
        def mock_chat(llm_messages, stream=False):
            last_msg = llm_messages[-1].get("Content", "") if llm_messages else ""
            
            # Capture script generation prompt
            if "播客脚本" in last_msg or "脚本" in last_msg:
                script_prompt_content.append(last_msg)
                # Return a script that includes some user content
                return {
                    "Choices": [{
                        "Message": {
                            "Content": "这是根据用户观点生成的播客脚本。用户提到了很多有趣的想法。"
                        }
                    }]
                }
            
            if "风格" in last_msg:
                return {
                    "Choices": [{
                        "Message": {
                            "Content": json.dumps({
                                "tone": "自然",
                                "vocabulary": [],
                                "expressions": [],
                                "sentence_style": "口语化"
                            }, ensure_ascii=False)
                        }
                    }]
                }
            
            return {
                "Choices": [{"Message": {"Content": "AI回复"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            # Add user opinions
            for opinion in opinions:
                agent.chat(session.session_id, opinion)
            
            # Verify key points were extracted
            assert len(session.key_points) >= 1, "Should have at least one key point"
            
            # Generate script
            result = agent.generate_script(session.session_id)
            
            # Property 7a: Script is non-empty
            assert "script" in result, "Result should have 'script' field"
            assert len(result["script"]) > 0, "Script should be non-empty"
            
            # Property 7b: Key points were included in the prompt
            assert len(script_prompt_content) > 0, "Script generation should have been called"
            prompt = script_prompt_content[0]
            
            # The prompt should contain key points section
            assert "用户核心观点" in prompt or "观点" in prompt, \
                "Script prompt should include user key points section"
            
            # Property 7c: style_applied is returned
            assert "style_applied" in result, "Result should have 'style_applied' field"


class TestSourceCitations:
    """
    **Feature: interview-podcast-mode, Property 8: Source citations in script**
    **Validates: Requirements 3.5, 8.3**
    
    For any session with materials, the generated script shall include 
    source citations for referenced materials.
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        num_materials=st.integers(min_value=1, max_value=5),
        material_types=st.lists(
            st.sampled_from(['url', 'document', 'topic']),
            min_size=1,
            max_size=5
        )
    )
    def test_script_includes_source_citations(self, num_materials, material_types):
        """
        **Feature: interview-podcast-mode, Property 8: Source citations in script**
        **Validates: Requirements 3.5, 8.3**
        
        Property: For any session with materials, generate_script returns 
        a sources list containing all added materials.
        """
        from unittest.mock import patch, MagicMock
        import json
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
            "bocha_base_url": "https://api.bocha.test",
            "bocha_api_id": "test_id",
            "bocha_api_key": "test_key",
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        # Track script generation prompt
        script_prompt_content = []
        
        def mock_chat(llm_messages, stream=False):
            last_msg = llm_messages[-1].get("Content", "") if llm_messages else ""
            
            if "播客脚本" in last_msg:
                script_prompt_content.append(last_msg)
                return {
                    "Choices": [{
                        "Message": {
                            "Content": "这是生成的脚本，引用了相关素材。"
                        }
                    }]
                }
            
            if "风格" in last_msg:
                return {
                    "Choices": [{
                        "Message": {
                            "Content": json.dumps({
                                "tone": "自然",
                                "vocabulary": [],
                                "expressions": [],
                                "sentence_style": "口语化"
                            }, ensure_ascii=False)
                        }
                    }]
                }
            
            # Default response for summaries and AI thoughts
            return {
                "Choices": [{"Message": {"Content": "这是摘要内容。"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client, \
             patch('pipeline.interview_agent.fetch_url_enhanced') as mock_fetch, \
             patch('pipeline.interview_agent.BochaClient') as mock_bocha_class:
            
            mock_client = MagicMock()
            mock_client.chat = mock_chat
            mock_get_client.return_value = mock_client
            
            mock_fetch.return_value = {
                "success": True,
                "text": "Fetched content",
                "status": 200
            }
            
            mock_bocha_instance = MagicMock()
            mock_bocha_instance.search.return_value = [
                {"title": "Result", "snippet": "Snippet", "url": "https://test.com"}
            ]
            mock_bocha_class.return_value = mock_bocha_instance
            
            # Add materials
            types_to_add = material_types[:num_materials]
            added_sources = []
            
            for i, mat_type in enumerate(types_to_add):
                content = f"Test content {i}"
                if mat_type == "url":
                    content = f"https://example{i}.com/article"
                
                result = agent.add_material(session.session_id, mat_type, content)
                added_sources.append(result["source"])
            
            # Verify materials were added
            assert len(session.materials) == len(types_to_add), \
                f"Expected {len(types_to_add)} materials, got {len(session.materials)}"
            
            # Generate script
            result = agent.generate_script(session.session_id)
            
            # Property 8a: sources list is returned
            assert "sources" in result, "Result should have 'sources' field"
            assert isinstance(result["sources"], list), "sources should be a list"
            
            # Property 8b: sources list has same count as materials
            assert len(result["sources"]) == len(session.materials), \
                f"Expected {len(session.materials)} sources, got {len(result['sources'])}"
            
            # Property 8c: Each source has required fields
            for source in result["sources"]:
                assert "type" in source, "Source should have 'type' field"
                assert "source" in source, "Source should have 'source' field"
                assert "summary" in source, "Source should have 'summary' field"
            
            # Property 8d: Materials were included in script prompt
            if script_prompt_content:
                prompt = script_prompt_content[0]
                assert "参考素材" in prompt or "素材" in prompt, \
                    "Script prompt should include materials section"


class TestErrorHandlingPreservesSession:
    """
    **Feature: interview-podcast-mode, Property 10: Error handling preserves session**
    **Validates: Requirements 7.6**
    
    For any API error during chat or material addition, the session data 
    shall remain intact and accessible.
    """
    
    def setup_method(self):
        """Clear sessions before each test."""
        clear_sessions()
    
    @settings(max_examples=100)
    @given(
        initial_messages=st.lists(
            st.text(min_size=10, max_size=200).filter(lambda x: x.strip()),
            min_size=1,
            max_size=5
        ),
        error_message=st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
    )
    def test_session_preserved_after_chat_error(self, initial_messages, error_message):
        """
        **Feature: interview-podcast-mode, Property 10: Error handling preserves session**
        **Validates: Requirements 7.6**
        
        Property: When an error occurs during chat, the session data 
        (messages, key_points, materials) remains intact.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        call_count = [0]
        
        def mock_chat_with_error(llm_messages, stream=False):
            call_count[0] += 1
            # First few calls succeed, then fail
            if call_count[0] <= len(initial_messages):
                return {
                    "Choices": [{"Message": {"Content": "AI回复"}}]
                }
            # Simulate error by raising exception
            raise Exception(f"Simulated LLM error: {error_message}")
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat_with_error
            mock_get_client.return_value = mock_client
            
            # Add initial messages successfully
            for msg in initial_messages:
                agent.chat(session.session_id, msg)
            
            # Record state before error
            messages_before = len(session.messages)
            key_points_before = len(session.key_points)
            materials_before = len(session.materials)
            
            # Attempt chat that will fail
            try:
                agent.chat(session.session_id, "This message will cause an error")
            except Exception:
                pass  # Expected error
            
            # Property 10a: Session still exists and is accessible
            retrieved_session = agent.get_session(session.session_id)
            assert retrieved_session is not None, "Session should still exist after error"
            
            # Property 10b: Session ID is unchanged
            assert retrieved_session.session_id == session.session_id, \
                "Session ID should be unchanged"
            
            # Property 10c: Previous messages are preserved
            # Note: The failed message might have been added before the error
            assert len(retrieved_session.messages) >= messages_before, \
                f"Messages should be preserved, had {messages_before}, now have {len(retrieved_session.messages)}"
            
            # Property 10d: Key points are preserved
            assert len(retrieved_session.key_points) >= key_points_before, \
                f"Key points should be preserved, had {key_points_before}, now have {len(retrieved_session.key_points)}"
            
            # Property 10e: Materials are preserved
            assert len(retrieved_session.materials) == materials_before, \
                f"Materials should be preserved, had {materials_before}, now have {len(retrieved_session.materials)}"
    
    @settings(max_examples=100)
    @given(
        initial_materials=st.lists(
            st.tuples(
                st.sampled_from(['url', 'document', 'topic']),
                st.text(min_size=10, max_size=100).filter(lambda x: x.strip())
            ),
            min_size=1,
            max_size=3
        )
    )
    def test_session_preserved_after_material_error(self, initial_materials):
        """
        **Feature: interview-podcast-mode, Property 10: Error handling preserves session**
        **Validates: Requirements 7.6**
        
        Property: When an error occurs during material addition, the session data 
        remains intact and previously added materials are preserved.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
            "bocha_base_url": "https://api.bocha.test",
            "bocha_api_id": "test_id",
            "bocha_api_key": "test_key",
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        material_call_count = [0]
        
        def mock_chat_for_summary(llm_messages, stream=False):
            material_call_count[0] += 1
            # First few calls succeed, then fail
            if material_call_count[0] <= len(initial_materials) * 2:  # 2 calls per material (summary + thoughts)
                return {
                    "Choices": [{"Message": {"Content": "摘要内容"}}]
                }
            raise Exception("Simulated LLM error during material processing")
        
        with patch.object(agent, '_get_llm_client') as mock_get_client, \
             patch('pipeline.interview_agent.fetch_url_enhanced') as mock_fetch, \
             patch('pipeline.interview_agent.BochaClient') as mock_bocha_class:
            
            mock_client = MagicMock()
            mock_client.chat = mock_chat_for_summary
            mock_get_client.return_value = mock_client
            
            mock_fetch.return_value = {
                "success": True,
                "text": "Fetched content",
                "status": 200
            }
            
            mock_bocha_instance = MagicMock()
            mock_bocha_instance.search.return_value = [
                {"title": "Result", "snippet": "Snippet", "url": "https://test.com"}
            ]
            mock_bocha_class.return_value = mock_bocha_instance
            
            # Add initial materials successfully
            for mat_type, content in initial_materials:
                if mat_type == "url":
                    content = f"https://example.com/{content.replace(' ', '-')[:30]}"
                agent.add_material(session.session_id, mat_type, content)
            
            # Record state before error
            materials_before = len(session.materials)
            material_ids_before = [m["id"] for m in session.materials]
            
            # Attempt to add material that will fail
            try:
                agent.add_material(session.session_id, "topic", "This will cause an error")
            except Exception:
                pass  # Expected error
            
            # Property 10a: Session still exists
            retrieved_session = agent.get_session(session.session_id)
            assert retrieved_session is not None, "Session should still exist after error"
            
            # Property 10b: Previously added materials are preserved
            assert len(retrieved_session.materials) >= materials_before, \
                f"Materials should be preserved, had {materials_before}, now have {len(retrieved_session.materials)}"
            
            # Property 10c: Material IDs are unchanged
            current_ids = [m["id"] for m in retrieved_session.materials[:materials_before]]
            assert current_ids == material_ids_before, \
                "Previously added material IDs should be unchanged"
    
    @settings(max_examples=50)
    @given(
        session_count=st.integers(min_value=2, max_value=5)
    )
    def test_error_in_one_session_does_not_affect_others(self, session_count):
        """
        **Feature: interview-podcast-mode, Property 10: Error handling preserves session**
        **Validates: Requirements 7.6**
        
        Property: An error in one session does not affect other sessions.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        
        # Create multiple sessions
        sessions = []
        for _ in range(session_count):
            sessions.append(agent.start_session())
        
        # Track which session is being used
        error_triggered = [False]
        
        def mock_chat_selective_error(llm_messages, stream=False):
            # Only fail when explicitly triggered
            if error_triggered[0]:
                error_triggered[0] = False  # Reset for next call
                raise Exception("Simulated error for first session")
            return {
                "Choices": [{"Message": {"Content": "AI回复"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat_selective_error
            mock_get_client.return_value = mock_client
            
            # Add message to first session (will succeed)
            agent.chat(sessions[0].session_id, "First message")
            
            # Add messages to other sessions
            for i in range(1, session_count):
                agent.chat(sessions[i].session_id, f"Message for session {i}")
            
            # Record state of other sessions
            other_session_states = []
            for i in range(1, session_count):
                other_session_states.append({
                    "id": sessions[i].session_id,
                    "message_count": len(sessions[i].messages)
                })
            
            # Cause error in first session
            error_triggered[0] = True
            try:
                agent.chat(sessions[0].session_id, "This will cause error")
            except Exception:
                pass
            
            # Property 10: Other sessions are unaffected
            for i, state in enumerate(other_session_states):
                session = agent.get_session(state["id"])
                assert session is not None, f"Session {i+1} should still exist"
                assert len(session.messages) == state["message_count"], \
                    f"Session {i+1} message count should be unchanged"
    
    @settings(max_examples=50)
    @given(
        messages_before_error=st.lists(
            st.text(min_size=10, max_size=100).filter(lambda x: x.strip()),
            min_size=1,
            max_size=3
        )
    )
    def test_session_recoverable_after_error(self, messages_before_error):
        """
        **Feature: interview-podcast-mode, Property 10: Error handling preserves session**
        **Validates: Requirements 7.6**
        
        Property: After an error, the session can continue to be used normally.
        """
        from unittest.mock import patch, MagicMock
        
        clear_sessions()
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 2000,
        }
        
        agent = InterviewAgent(cfg=mock_cfg)
        session = agent.start_session()
        
        call_count = [0]
        error_on_call = len(messages_before_error) + 1  # Error on first call after initial messages
        
        def mock_chat_recoverable(llm_messages, stream=False):
            call_count[0] += 1
            if call_count[0] == error_on_call:
                raise Exception("Temporary error")
            return {
                "Choices": [{"Message": {"Content": "AI回复"}}]
            }
        
        with patch.object(agent, '_get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat = mock_chat_recoverable
            mock_get_client.return_value = mock_client
            
            # Add initial messages
            for msg in messages_before_error:
                agent.chat(session.session_id, msg)
            
            messages_after_initial = len(session.messages)
            
            # Cause error
            try:
                agent.chat(session.session_id, "Error message")
            except Exception:
                pass
            
            # Property 10a: Session can continue after error
            result = agent.chat(session.session_id, "Recovery message")
            
            assert "reply" in result, "Should get reply after recovery"
            assert len(result["reply"]) > 0, "Reply should be non-empty"
            
            # Property 10b: New messages are added after recovery
            # We expect: initial messages + (possibly partial error message) + recovery message + AI reply
            assert len(session.messages) > messages_after_initial, \
                "New messages should be added after recovery"
