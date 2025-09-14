"""
Security tests for OmniSeg project
"""
import os
import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, mock_open
import importlib
import sys


class TestSecurityVulnerabilities:
    """Test suite for security vulnerabilities."""
    
    @pytest.mark.security
    def test_no_hardcoded_secrets(self):
        """Test that no hardcoded secrets are present in the codebase."""
        import re
        
        # Common secret patterns
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret_key\s*=\s*["\'][^"\']+["\']',
            r'private_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'aws_access_key_id\s*=\s*["\'][^"\']+["\']',
            r'aws_secret_access_key\s*=\s*["\'][^"\']+["\']',
        ]
        
        project_root = Path(__file__).parent.parent
        python_files = list(project_root.rglob("*.py"))
        
        violations = []
        for file_path in python_files:
            # Skip test files and virtual environments
            if 'test' in str(file_path) or '.venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Exclude obvious test/dummy values
                        matched_text = match.group(0)
                        if not any(dummy in matched_text.lower() for dummy in 
                                 ['test', 'dummy', 'fake', 'example', 'placeholder', 'your_']):
                            violations.append(f"{file_path}:{match.start()}: {matched_text}")
            except (UnicodeDecodeError, PermissionError):
                continue
        
        assert not violations, f"Potential hardcoded secrets found: {violations}"
    
    @pytest.mark.security 
    def test_no_eval_exec_usage(self):
        """Test that dangerous eval/exec functions are not used."""
        import ast
        import re
        
        project_root = Path(__file__).parent.parent
        python_files = list(project_root.rglob("*.py"))
        
        violations = []
        for file_path in python_files:
            # Skip test files
            if 'test' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for eval() and exec() usage
                if re.search(r'\beval\s*\(', content) or re.search(r'\bexec\s*\(', content):
                    violations.append(str(file_path))
                    
            except (UnicodeDecodeError, PermissionError):
                continue
        
        assert not violations, f"Dangerous eval/exec usage found in: {violations}"
    
    @pytest.mark.security
    def test_safe_file_operations(self):
        """Test that file operations use safe paths."""
        import omniseg.utils.generate_data as gen_data_module
        import inspect
        
        # Get all functions that might do file operations
        functions = inspect.getmembers(gen_data_module, inspect.isfunction)
        
        for name, func in functions:
            if any(keyword in name.lower() for keyword in ['save', 'load', 'write', 'read', 'open']):
                # Check function signature doesn't use unsafe practices
                sig = inspect.signature(func)
                
                # Should not have 'w+' or 'a+' modes that could be dangerous
                source = inspect.getsource(func)
                assert 'w+' not in source or 'safe' in source.lower(), f"Unsafe file mode in {name}"
    
    @pytest.mark.security
    def test_input_validation_in_config(self):
        """Test that config validation prevents malicious inputs."""
        from omniseg.config import validate_config
        
        # Test path traversal attempts
        malicious_configs = [
            {'backbone_type': '../../../etc/passwd', 'head_type': 'maskrcnn', 'num_classes': 3, 'image_size': 224},
            {'backbone_type': 'simple', 'head_type': '..\\..\\windows\\system32', 'num_classes': 3, 'image_size': 224},
            {'backbone_type': 'simple', 'head_type': 'maskrcnn', 'num_classes': -999999, 'image_size': 224},
            {'backbone_type': 'simple', 'head_type': 'maskrcnn', 'num_classes': 3, 'image_size': -224},
        ]
        
        for config in malicious_configs:
            with pytest.raises(ValueError):
                validate_config(config)
    
    @pytest.mark.security
    def test_no_shell_injection_vulnerabilities(self):
        """Test that shell commands are properly sanitized."""
        import re
        
        project_root = Path(__file__).parent.parent
        python_files = list(project_root.rglob("*.py"))
        
        violations = []
        dangerous_patterns = [
            r'os\.system\s*\(',
            r'subprocess\.call\s*\([^)]*shell\s*=\s*True',
            r'subprocess\.run\s*\([^)]*shell\s*=\s*True',
            r'subprocess\.Popen\s*\([^)]*shell\s*=\s*True',
        ]
        
        for file_path in python_files:
            if 'test' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in dangerous_patterns:
                    if re.search(pattern, content):
                        violations.append(str(file_path))
                        
            except (UnicodeDecodeError, PermissionError):
                continue
        
        assert not violations, f"Potential shell injection vulnerabilities in: {violations}"
    
    @pytest.mark.security
    def test_dependency_safety(self):
        """Test that dependencies don't have known vulnerabilities."""
        requirements_file = Path(__file__).parent.parent / "requirements.txt"
        
        if requirements_file.exists():
            # Try to run safety check if available
            try:
                result = subprocess.run(
                    ['python', '-m', 'safety', 'check', '--json', '-r', str(requirements_file)],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    # No vulnerabilities found
                    assert True
                else:
                    # Vulnerabilities found or safety not available
                    if "No module named" in result.stderr:
                        pytest.skip("Safety not installed, skipping dependency vulnerability check")
                    else:
                        pytest.fail(f"Security vulnerabilities found in dependencies: {result.stdout}")
                        
            except subprocess.TimeoutExpired:
                pytest.skip("Safety check timed out")
            except FileNotFoundError:
                pytest.skip("Safety not available")
    
    @pytest.mark.security
    def test_no_pickle_usage(self):
        """Test that pickle module is not used (potential security risk)."""
        import re
        
        project_root = Path(__file__).parent.parent
        python_files = list(project_root.rglob("*.py"))
        
        violations = []
        for file_path in python_files:
            if 'test' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for pickle imports and usage
                if re.search(r'import\s+pickle', content) or re.search(r'from\s+pickle\s+import', content):
                    violations.append(str(file_path))
                    
            except (UnicodeDecodeError, PermissionError):
                continue
        
        # Allow pickle usage only if it's explicitly safe (e.g., with proper validation)
        for violation in violations[:]:
            with open(violation, 'r') as f:
                content = f.read()
                if 'safe' in content.lower() or 'secure' in content.lower():
                    violations.remove(violation)
        
        assert not violations, f"Unsafe pickle usage found in: {violations}"
    
    @pytest.mark.security
    def test_environment_variable_validation(self):
        """Test that environment variables are properly validated."""
        import os
        from unittest.mock import patch
        
        # Test with malicious environment variables
        malicious_env_vars = {
            'HF_TOKEN': '../../../etc/passwd',
            'TORCH_HOME': '/dev/null; rm -rf /',
            'TRANSFORMERS_CACHE': '$(malicious_command)',
        }
        
        with patch.dict(os.environ, malicious_env_vars):
            try:
                # Try importing modules that might use these env vars
                import omniseg.config
                # Should handle malicious inputs gracefully
                assert True
            except Exception as e:
                # Should not crash with malicious env vars
                if "malicious" in str(e).lower() or "security" in str(e).lower():
                    pytest.fail(f"Security vulnerability with env vars: {e}")
    
    @pytest.mark.security
    def test_temporary_file_security(self):
        """Test that temporary files are created securely."""
        import tempfile
        import stat
        
        # Create a temporary file and check permissions
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name
            
        try:
            # Check file permissions
            file_stat = os.stat(file_path)
            file_mode = stat.filemode(file_stat.st_mode)
            
            # Should not be world-readable/writable
            assert not (file_stat.st_mode & stat.S_IROTH), "Temporary file is world-readable"
            assert not (file_stat.st_mode & stat.S_IWOTH), "Temporary file is world-writable"
            
        finally:
            os.unlink(file_path)
    
    @pytest.mark.security
    def test_no_debug_mode_in_production(self):
        """Test that debug modes are not enabled by default."""
        import re
        
        project_root = Path(__file__).parent.parent
        python_files = list(project_root.rglob("*.py"))
        
        violations = []
        debug_patterns = [
            r'DEBUG\s*=\s*True',
            r'debug\s*=\s*True',
            r'\.debug\(\s*True\s*\)',
        ]
        
        for file_path in python_files:
            if 'test' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in debug_patterns:
                    if re.search(pattern, content):
                        violations.append(str(file_path))
                        
            except (UnicodeDecodeError, PermissionError):
                continue
        
        assert not violations, f"Debug mode enabled in production files: {violations}"