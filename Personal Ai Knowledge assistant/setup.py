"""
Setup script for Personal AI Knowledge Assistant
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "embeddings", 
        "logs",
        "data/emails",
        "data/notes",
        "data/learning",
        "data/education",
        "data/work",
        "data/resume",
        "data/other"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_env_file():
    """Create .env file from template"""
    if not os.path.exists(".env"):
        if os.path.exists("env_template.txt"):
            with open("env_template.txt", "r") as template:
                content = template.read()
            
            with open(".env", "w") as env_file:
                env_file.write(content)
            
            print("Created .env file from template")
            print("Please edit .env file and add your API keys")
        else:
            print("Warning: env_template.txt not found")
    else:
        print(".env file already exists")

def main():
    """Main setup function"""
    print("Setting up Personal AI Knowledge Assistant...")
    
    # Create directories
    create_directories()
    
    # Create environment file
    create_env_file()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Edit .env file with your API keys")
    print("3. Run the application: streamlit run app.py")
    print("\nFor Groq API key, visit: https://console.groq.com/")

if __name__ == "__main__":
    main()
