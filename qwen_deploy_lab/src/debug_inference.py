import sys
sys.path.append("src")
from inference import generate_response

print(generate_response("你好", max_tokens=20))