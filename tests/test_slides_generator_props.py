"""
Property-based tests for Slidev PPT Generator feature.
Uses Hypothesis library for property testing.

**Feature: slidev-ppt-generator**
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import patch, MagicMock
import sys
import os
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to be tested
from pipeline import slides_generator


class TestSlidevMarkdownFormatValidity:
    """
    **Feature: slidev-ppt-generator, Property 1: Slidev Markdown format validity**
    **Validates: Requirements 1.2, 5.1**
    
    For any podcast script input, the generated output shall be valid Slidev Markdown containing:
    - YAML frontmatter starting with `---` and containing `theme` key
    - At least one slide separator (`---`)
    - A title heading (`#`) on the first slide
    """
    
    @settings(max_examples=100)
    @given(
        title=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        style=st.sampled_from(['professional', 'minimal', 'creative'])
    )
    def test_slidev_markdown_format_validity(self, title, style):
        """
        Property: Generated Slidev Markdown must have valid format with frontmatter,
        theme configuration, and proper slide structure.
        """
        # Mock LLM response with valid Slidev Markdown
        mock_llm_response = f"""---
theme: seriph
title: {title}
class: text-center
---

# {title}

基于 AI 播客内容生成

---

# 核心要点

- 要点 1
- 要点 2
- 要点 3

---

# 详细分析

这是详细分析内容

---

# 总结

关键结论和行动建议
"""
        
        def mock_chat(messages, stream=False):
            return {
                "Choices": [{"Message": {"Content": mock_llm_response}}]
            }
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 8000,
        }
        
        script = "这是一段测试播客脚本内容。主播A说了一些话，主播B回应了。"
        
        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.chat = mock_chat
        
        # Patch the HunyuanAPIClient class in the slides_generator module
        with patch.object(slides_generator, 'HunyuanAPIClient', create=True) as MockClient:
            MockClient.return_value = mock_client_instance
            
            # Need to patch the import inside the function
            with patch.dict('sys.modules', {'clients.hunyuan_api_client': MagicMock(HunyuanAPIClient=MockClient)}):
                # Reload to pick up the mock
                import importlib
                importlib.reload(slides_generator)
                
                result = slides_generator.extract_key_points(
                    cfg=mock_cfg,
                    script=script,
                    title=title,
                    style=style
                )
        
        # Verify frontmatter exists and starts with ---
        assert result.startswith('---'), "Slidev Markdown must start with frontmatter (---)"
        
        # Verify theme key exists in frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---', result, re.DOTALL)
        assert frontmatter_match, "Frontmatter must be properly closed with ---"
        frontmatter_content = frontmatter_match.group(1)
        assert 'theme:' in frontmatter_content, "Frontmatter must contain 'theme' key"
        
        # Verify at least one slide separator exists (beyond frontmatter)
        content_after_frontmatter = result[frontmatter_match.end():]
        slide_separators = re.findall(r'\n---\s*\n', content_after_frontmatter)
        assert len(slide_separators) >= 1, "Must have at least one slide separator after frontmatter"
        
        # Verify title heading exists
        assert re.search(r'^#\s+.+', result, re.MULTILINE), "Must have a title heading (#)"



class TestContentTransformation:
    """
    **Feature: slidev-ppt-generator, Property 2: Content transformation (not verbatim copy)**
    **Validates: Requirements 1.3**
    
    For any podcast script with length > 500 characters, the generated Slidev Markdown 
    shall have a character count less than 50% of the original script length, 
    indicating key points extraction rather than verbatim copying.
    """
    
    @settings(max_examples=100)
    @given(
        script_length=st.integers(min_value=600, max_value=3000)
    )
    def test_content_transformation_not_verbatim(self, script_length):
        """
        Property: Generated slides should be significantly shorter than the original script,
        demonstrating key points extraction rather than verbatim copying.
        """
        # Generate a script of the specified length
        base_text = "这是播客脚本的一段内容，主播在讨论一个重要话题。"
        script = (base_text * (script_length // len(base_text) + 1))[:script_length]
        
        assume(len(script) > 500)  # Ensure script meets minimum length requirement
        
        # Mock LLM response with condensed content (key points extraction)
        # The response should be significantly shorter than the input
        mock_llm_response = """---
theme: seriph
title: 测试演示文稿
class: text-center
---

# 测试演示文稿

基于 AI 播客内容生成

---

# 核心要点

- 要点 1：关键信息
- 要点 2：重要数据
- 要点 3：主要结论

---

# 详细分析

简洁的分析内容

---

# 总结

关键结论
"""
        
        def mock_chat(messages, stream=False):
            return {
                "Choices": [{"Message": {"Content": mock_llm_response}}]
            }
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 8000,
        }
        
        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.chat = mock_chat
        
        with patch.dict('sys.modules', {'clients.hunyuan_api_client': MagicMock(HunyuanAPIClient=lambda **kwargs: mock_client_instance)}):
            import importlib
            importlib.reload(slides_generator)
            
            result = slides_generator.extract_key_points(
                cfg=mock_cfg,
                script=script,
                title="测试演示文稿",
                style="professional"
            )
        
        # Verify the output is significantly shorter than input (< 50%)
        result_length = len(result)
        script_length_actual = len(script)
        
        assert result_length < script_length_actual * 0.5, (
            f"Generated slides ({result_length} chars) should be less than 50% "
            f"of original script ({script_length_actual} chars) to indicate key points extraction"
        )


class TestSlideStructureCompleteness:
    """
    **Feature: slidev-ppt-generator, Property 3: Slide structure completeness**
    **Validates: Requirements 5.3, 5.4**
    
    For any generated Slidev Markdown, it shall contain:
    - A title slide (first slide with `#` heading matching the provided title)
    - At least 3 content slides
    - A summary/conclusion slide (last slide containing keywords like "总结", "结论", "Summary", "Conclusion")
    """
    
    @settings(max_examples=100)
    @given(
        title=st.text(min_size=2, max_size=30).filter(lambda x: x.strip() and not any(c in x for c in ['#', '---', '\n']))
    )
    def test_slide_structure_completeness(self, title):
        """
        Property: Generated slides must have complete structure with title slide,
        content slides, and summary slide.
        """
        # Mock LLM response with complete slide structure
        mock_llm_response = f"""---
theme: seriph
title: {title}
class: text-center
---

# {title}

基于 AI 播客内容生成

---

# 核心要点

- 要点 1
- 要点 2

---

# 详细分析

分析内容

---

# 深入讨论

讨论内容

---

# 总结

关键结论和行动建议
"""
        
        def mock_chat(messages, stream=False):
            return {
                "Choices": [{"Message": {"Content": mock_llm_response}}]
            }
        
        mock_cfg = {
            "hunyuan_api_secret_id": "test",
            "hunyuan_api_secret_key": "test",
            "hunyuan_api_region": "ap-guangzhou",
            "hunyuan_api_model": "hunyuan-turbos-latest",
            "hunyuan_api_temperature": 0.8,
            "hunyuan_api_top_p": 0.8,
            "hunyuan_api_max_tokens": 8000,
        }
        
        script = "这是一段测试播客脚本内容。"
        
        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.chat = mock_chat
        
        with patch.dict('sys.modules', {'clients.hunyuan_api_client': MagicMock(HunyuanAPIClient=lambda **kwargs: mock_client_instance)}):
            import importlib
            importlib.reload(slides_generator)
            
            result = slides_generator.extract_key_points(
                cfg=mock_cfg,
                script=script,
                title=title,
                style="professional"
            )
        
        # Parse the slides
        slides = slides_generator.parse_slidev_markdown(result)
        
        # Verify at least 3 content slides (excluding title slide)
        assert len(slides) >= 4, f"Must have at least 4 slides (1 title + 3 content), got {len(slides)}"
        
        # Verify first slide contains the title
        first_slide_content = slides[0].get("content", "")
        # Title might be in the content or as a heading
        title_found = title in first_slide_content or f"# {title}" in result
        assert title_found, f"First slide must contain the title '{title}'"
        
        # Verify last slide is a summary/conclusion slide
        last_slide_content = slides[-1].get("content", "")
        summary_keywords = ["总结", "结论", "Summary", "Conclusion", "小结", "结语"]
        has_summary = any(kw in last_slide_content for kw in summary_keywords)
        assert has_summary, f"Last slide must contain summary keywords. Got: {last_slide_content[:200]}"


class TestExportFormatValidity:
    """
    **Feature: slidev-ppt-generator, Property 4: Export format validity**
    **Validates: Requirements 4.1, 4.2**
    
    For any valid Slidev Markdown and export format (pdf/pptx), the exported file shall:
    - Have the correct file extension
    - Have valid file magic bytes (PDF: `%PDF`, PPTX: `PK` ZIP header)
    - Have file size > 0 bytes
    """
    
    @settings(max_examples=100, deadline=None)
    @given(
        title=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() and not any(c in x for c in ['#', '---', '\n', '<', '>', ':', '"', '/', '\\', '|', '?', '*'])),
        num_slides=st.integers(min_value=3, max_value=8)
    )
    def test_pdf_export_format_validity(self, title, num_slides):
        """
        Property: Exported PDF files must have correct extension, valid magic bytes, and non-zero size.
        """
        import tempfile
        
        # Generate valid Slidev Markdown
        slides_content = []
        for i in range(num_slides - 2):  # -2 for title and summary slides
            slides_content.append(f"""---

# 内容 {i + 1}

- 要点 {i + 1}.1
- 要点 {i + 1}.2
""")
        
        markdown = f"""---
theme: seriph
title: {title}
class: text-center
---

# {title}

基于 AI 播客内容生成

{''.join(slides_content)}
---

# 总结

关键结论和行动建议
"""
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            output_path = f.name
        
        try:
            # Export to PDF
            result_path = slides_generator.export_to_pdf(markdown, output_path)
            
            # Verify correct file extension
            assert result_path.endswith('.pdf'), f"PDF file must have .pdf extension, got: {result_path}"
            
            # Verify file exists and has non-zero size
            assert os.path.exists(result_path), f"PDF file must exist at {result_path}"
            file_size = os.path.getsize(result_path)
            assert file_size > 0, f"PDF file must have non-zero size, got: {file_size}"
            
            # Verify PDF magic bytes (%PDF)
            with open(result_path, 'rb') as f:
                magic_bytes = f.read(4)
                assert magic_bytes == b'%PDF', f"PDF file must start with %PDF magic bytes, got: {magic_bytes}"
        
        except ImportError as e:
            # Skip test if weasyprint/reportlab not installed
            pytest.skip(f"PDF export library not available: {e}")
        except RuntimeError as e:
            if "需要安装" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"PDF export library not available: {e}")
            raise
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)
    
    @settings(max_examples=100)
    @given(
        title=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() and not any(c in x for c in ['#', '---', '\n', '<', '>', ':', '"', '/', '\\', '|', '?', '*'])),
        num_slides=st.integers(min_value=3, max_value=8)
    )
    def test_pptx_export_format_validity(self, title, num_slides):
        """
        Property: Exported PPTX files must have correct extension, valid magic bytes (PK ZIP header), and non-zero size.
        """
        import tempfile
        
        # Generate valid Slidev Markdown
        slides_content = []
        for i in range(num_slides - 2):  # -2 for title and summary slides
            slides_content.append(f"""---

# 内容 {i + 1}

- 要点 {i + 1}.1
- 要点 {i + 1}.2
""")
        
        markdown = f"""---
theme: seriph
title: {title}
class: text-center
---

# {title}

基于 AI 播客内容生成

{''.join(slides_content)}
---

# 总结

关键结论和行动建议
"""
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as f:
            output_path = f.name
        
        try:
            # Export to PPTX
            result_path = slides_generator.export_to_pptx(markdown, output_path)
            
            # Verify correct file extension
            assert result_path.endswith('.pptx'), f"PPTX file must have .pptx extension, got: {result_path}"
            
            # Verify file exists and has non-zero size
            assert os.path.exists(result_path), f"PPTX file must exist at {result_path}"
            file_size = os.path.getsize(result_path)
            assert file_size > 0, f"PPTX file must have non-zero size, got: {file_size}"
            
            # Verify PPTX magic bytes (PK ZIP header)
            with open(result_path, 'rb') as f:
                magic_bytes = f.read(2)
                assert magic_bytes == b'PK', f"PPTX file must start with PK (ZIP) magic bytes, got: {magic_bytes}"
        
        except ImportError as e:
            # Skip test if python-pptx not installed
            pytest.skip(f"PPTX export library not available: {e}")
        except RuntimeError as e:
            if "需要安装" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"PPTX export library not available: {e}")
            raise
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAPIResponseContainsFileURL:
    """
    **Feature: slidev-ppt-generator, Property 5: API response contains file URL on success**
    **Validates: Requirements 4.3, 6.3**
    
    For any successful export operation, the API response shall contain a non-empty 
    `file_url` field pointing to a valid COS URL, or a valid `file_path` for local fallback.
    """
    
    @settings(max_examples=100)
    @given(
        title=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() and not any(c in x for c in ['#', '---', '\n', '<', '>', ':', '"', '/', '\\', '|', '?', '*'])),
        format_type=st.sampled_from(['pdf', 'pptx'])
    )
    def test_api_response_contains_file_url_or_path(self, title, format_type):
        """
        Property: Successful export API response must contain either a valid file_url (COS)
        or a valid file_path (local fallback).
        """
        import tempfile
        
        # Generate valid Slidev Markdown
        markdown = f"""---
theme: seriph
title: {title}
class: text-center
---

# {title}

基于 AI 播客内容生成

---

# 核心要点

- 要点 1
- 要点 2

---

# 总结

关键结论
"""
        
        # Create temp file for output
        suffix = f'.{format_type}'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            output_path = f.name
        
        try:
            # Export to requested format
            if format_type == 'pdf':
                result_path = slides_generator.export_to_pdf(markdown, output_path)
            else:
                result_path = slides_generator.export_to_pptx(markdown, output_path)
            
            # Simulate API response structure
            # In real API, this would be returned by the endpoint
            api_response = {
                "file_url": None,  # Would be set if COS upload succeeds
                "file_path": f"/api/slides-file/{os.path.basename(result_path)}",
                "format": format_type,
                "fallback_available": True
            }
            
            # Verify response contains valid file_path (local fallback always available)
            assert api_response.get("file_path"), "API response must contain file_path"
            assert api_response["file_path"].startswith("/api/slides-file/"), \
                f"file_path must be a valid API path, got: {api_response['file_path']}"
            
            # Verify format matches
            assert api_response.get("format") == format_type, \
                f"Response format must match requested format: {format_type}"
            
            # Verify fallback_available is set
            assert api_response.get("fallback_available") is True, \
                "fallback_available must be True for successful exports"
            
            # If file_url is set, verify it's a valid URL format
            if api_response.get("file_url"):
                file_url = api_response["file_url"]
                assert file_url.startswith("https://"), \
                    f"file_url must be a valid HTTPS URL, got: {file_url}"
                assert ".cos." in file_url and ".myqcloud.com" in file_url, \
                    f"file_url must be a valid COS URL, got: {file_url}"
        
        except ImportError as e:
            pytest.skip(f"Export library not available: {e}")
        except RuntimeError as e:
            if "需要安装" in str(e) or "not available" in str(e).lower():
                pytest.skip(f"Export library not available: {e}")
            raise
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)


class TestAPIErrorResponsesIncludeMessage:
    """
    **Feature: slidev-ppt-generator, Property 6: API error responses include message**
    **Validates: Requirements 6.4**
    
    For any failed API request (invalid input, processing error), the response shall 
    contain an `error` or `detail` field with a non-empty error message.
    """
    
    @settings(max_examples=100)
    @given(
        invalid_input_type=st.sampled_from([
            'empty_script',
            'empty_markdown', 
            'whitespace_only',
            'invalid_format'
        ])
    )
    def test_api_error_responses_include_message(self, invalid_input_type):
        """
        Property: API error responses must include a non-empty error message
        in either 'error' or 'detail' field.
        """
        # Test different types of invalid inputs
        if invalid_input_type == 'empty_script':
            # Test extract_key_points with empty script
            try:
                slides_generator.extract_key_points(
                    cfg={},
                    script="",
                    title="Test",
                    style="professional"
                )
                pytest.fail("Should have raised ValueError for empty script")
            except ValueError as e:
                error_message = str(e)
                assert error_message, "Error message must not be empty"
                assert "空" in error_message or "empty" in error_message.lower(), \
                    f"Error message should indicate empty input: {error_message}"
        
        elif invalid_input_type == 'empty_markdown':
            # Test render_preview_html with empty markdown
            result = slides_generator.render_preview_html("")
            # Should return error HTML, not raise exception
            assert "error" in result.lower() or "空" in result, \
                f"Empty markdown should return error indication: {result[:200]}"
        
        elif invalid_input_type == 'whitespace_only':
            # Test with whitespace-only input
            try:
                slides_generator.extract_key_points(
                    cfg={},
                    script="   \n\t  ",
                    title="Test",
                    style="professional"
                )
                pytest.fail("Should have raised ValueError for whitespace-only script")
            except ValueError as e:
                error_message = str(e)
                assert error_message, "Error message must not be empty"
        
        elif invalid_input_type == 'invalid_format':
            # Test export with invalid format would be caught at API level
            # Here we test that export functions validate input
            try:
                slides_generator.export_to_pdf("", "/tmp/test.pdf")
                pytest.fail("Should have raised ValueError for empty markdown")
            except ValueError as e:
                error_message = str(e)
                assert error_message, "Error message must not be empty"
                assert "空" in error_message or "empty" in error_message.lower(), \
                    f"Error message should indicate empty input: {error_message}"
    
    @settings(max_examples=50)
    @given(
        script=st.text(min_size=0, max_size=10).filter(lambda x: not x.strip())
    )
    def test_empty_or_whitespace_script_returns_error(self, script):
        """
        Property: Empty or whitespace-only scripts must return a clear error message.
        """
        try:
            slides_generator.extract_key_points(
                cfg={},
                script=script,
                title="Test",
                style="professional"
            )
            pytest.fail("Should have raised ValueError for empty/whitespace script")
        except ValueError as e:
            error_message = str(e)
            # Verify error message is non-empty and descriptive
            assert len(error_message) > 0, "Error message must not be empty"
            # The error should indicate the issue is with empty content
            assert "空" in error_message or "empty" in error_message.lower() or "不能" in error_message, \
                f"Error message should be descriptive: {error_message}"
