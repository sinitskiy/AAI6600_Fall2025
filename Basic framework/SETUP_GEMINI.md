# Gemini AI Setup Instructions

This guide will help you set up the Google Gemini API integration for the Mental Health Chatbot Pipeline.

## Prerequisites

- Python 3.8 or higher
- Internet connection
- Google account

##  Step 1: Get Your Gemini API Key

### Option A: Using Google AI Studio (Recommended - Free)

1. **Visit Google AI Studio**
   - Go to [https://aistudio.google.com/](https://aistudio.google.com/)
   - Click "Get API Key" in the top right corner

2. **Sign in to your Google Account**
   - Use your existing Google account or create a new one

3. **Create a new API Key**
   - Click "Create API Key"
   - Select "Create API key in new project" (recommended)
   - Your API key will be generated instantly

4. **Copy your API Key**
   - **Important**: Copy this key immediately and store it securely

### Option B: Using Google Cloud Console (Advanced)

1. **Go to Google Cloud Console**
   - Visit [https://console.cloud.google.com/](https://console.cloud.google.com/)

2. **Create or Select a Project**
   - Create a new project or select an existing one

3. **Enable the Gemini API**
   - Go to "APIs & Services" > "Library"
   - Search for "Generative AI API" or "Gemini API"
   - Click "Enable"

4. **Create Credentials**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the generated API key

## ⚙️ Step 2: Create the Configuration File

1. **Navigate to the project directory**
   ```bash
   cd "/path/to/AAI6600_Fall2025/Basic framework"
   ```

2. **Create config.json file**
   
   **Option A: Using a text editor**
   ```bash
   nano config.json
   ```
   
   **Option B: Using VS Code**
   ```bash
   code config.json
   ```

3. **Add your API key to the file**
   Copy and paste the following content, replacing `YOUR_ACTUAL_API_KEY_HERE` with your real API key:

   ```json
   {
     "GEMINI_API_KEY": "YOUR_ACTUAL_API_KEY_HERE"
   }
   ```

   **Example with a real key:**
   ```json
   {
     "GEMINI_API_KEY": "AIzaSyC2Bk9H4e5vWjl6qW8ELpcMG808-zze2po"
   }
   ```

4. **Save the file**
   - Make sure the file is saved as `config.json` (not `config.json.txt`)
   - The file should be in the same directory as `chatbot_pipeline.py`

## 🔒 Step 3: Secure Your API Key

### Important Security Notes:

⚠️ **DO NOT commit config.json to Git!** This file contains your private API key.

1. **Check if config.json is ignored**
   ```bash
   git status
   ```
   - You should see `config.json` listed under "Untracked files"
   - This means it won't be committed to the repository ✅

2. **If config.json appears as a tracked file:**
   ```bash
   # Add it to .gitignore
   echo "config.json" >> .gitignore
   git add .gitignore
   git commit -m "Add config.json to gitignore"
   ```

3. **Never share your API key:**
   - Don't post it in chat messages
   - Don't include it in screenshots
   - Don't put it in public repositories

## Step 4: Install Required Dependencies

Make sure you have the required Python packages:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install required packages
pip install google-generativeai>=0.3.0 requests pandas numpy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## 🧪 Step 5: Test the Setup

1. **Run the chatbot pipeline**
   ```bash
   python chatbot_pipeline.py
   ```

2. **Expected behavior:**
   - The program should start successfully
   - You'll see the welcome message
   - Gemini should ask follow-up questions like "What city and state are you located in?"
   - No error messages about missing API keys

3. **If you see errors:**
   - Check that `config.json` is in the correct directory
   - Verify your API key is correctly formatted (no extra spaces)
   - Ensure you have internet connection

## Troubleshooting

### Error: "Could not read Gemini API key from config.json"
- **Solution**: Make sure `config.json` exists in the same folder as `chatbot_pipeline.py`
- Check the file name is exactly `config.json` (case-sensitive)

### Error: "GEMINI_API_KEY not found in config.json"
- **Solution**: Check the JSON format is correct
- Make sure the key name is exactly `"GEMINI_API_KEY"`
- Verify the JSON syntax (commas, quotes, brackets)

### Error: "Gemini API request failed"
- **Solution**: Verify your API key is valid
- Check your internet connection
- Make sure you haven't exceeded API quotas (Google AI Studio has generous free limits)

### Error: "Invalid JSON format"
- **Solution**: Validate your JSON syntax
- Use a JSON validator like [jsonlint.com](https://jsonlint.com/)
- Common issues: missing quotes, extra commas, wrong brackets

## API Usage and Limits

### Google AI Studio (Free Tier):
- **Free quota**: 60 requests per minute
- **Perfect for**: Development and testing
- **Cost**: $0

### Google Cloud (Pay-as-you-use):
- **Pricing**: Very affordable (typically $0.001-0.01 per request)
- **No request limits**
- **Perfect for**: Production deployment

## Support

If you encounter issues:

1. **Check the console output** - Error messages usually explain the problem
2. **Verify your API key** - Test it at [Google AI Studio](https://aistudio.google.com/)
3. **Check file permissions** - Make sure you can read `config.json`
4. **Review this guide** - Double-check each step

## Quick Setup Checklist

- [ ] Created Google account
- [ ] Generated Gemini API key from Google AI Studio
- [ ] Created `config.json` file in correct directory
- [ ] Added API key to config.json with correct JSON format
- [ ] Verified config.json is not tracked by Git
- [ ] Installed required Python packages
- [ ] Successfully ran `python chatbot_pipeline.py`
- [ ] Gemini responds with follow-up questions

##  Success

Once you see Gemini asking questions like "What city and state are you located in?", you've successfully integrated the AI! The chatbot can now:

- Understand user mental health needs
- Ask intelligent follow-up questions
- Gather location and insurance information
- Provide personalized facility recommendations