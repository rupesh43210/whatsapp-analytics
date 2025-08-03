# 📱 WhatsApp Chat Analytics Tool

A comprehensive Python tool that generates detailed analytics and beautiful visualizations from WhatsApp chat export files.

## ✨ Features

- **🔍 Robust Parsing**: Handles various WhatsApp export formats automatically
- **📊 Comprehensive Analytics**: 
  - Message statistics by participant
  - Time-based activity patterns
  - Word frequency analysis
  - Emoji usage statistics
  - Sentiment analysis
  - Response time analysis
  - Media vs text message breakdown
- **🎨 Amazing Visualizations**:
  - Interactive Plotly dashboard with 12+ charts
  - Word cloud generation
  - Responsive HTML report
- **📱 Export Format Support**:
  - DD/MM/YY, HH:MM format
  - MM/DD/YY, HH:MM AM/PM format
  - DD.MM.YY, HH:MM format
  - YYYY-MM-DD HH:MM:SS format
  - Bracketed timestamp formats
  - And more!

## 🚀 Quick Start

### Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

```bash
python whatsapp_analyzer.py path/to/your/chat_export.txt
```

Or specify custom output directory:
```bash
python whatsapp_analyzer.py path/to/your/chat_export.txt --output my_analysis
```

## 📁 How to Export WhatsApp Chat

### On Android:
1. Open WhatsApp
2. Go to the chat you want to analyze
3. Tap the three dots (⋮) → More → Export chat
4. Choose "Without Media" for faster processing
5. Save the .txt file

### On iPhone:
1. Open WhatsApp
2. Go to the chat you want to analyze
3. Tap the contact/group name at the top
4. Scroll down and tap "Export Chat"
5. Choose "Without Media"
6. Save the .txt file

## 📊 What You'll Get

The tool generates:

1. **📈 Interactive HTML Report** (`whatsapp_analytics_report.html`)
   - Beautiful dashboard with 12+ interactive charts
   - Key insights and statistics
   - Participant breakdown
   - Time-based analysis
   - Word and emoji analysis

2. **☁️ Word Cloud** (`wordcloud.png`)
   - Visual representation of most used words

3. **📋 Raw Analytics Data** (`analytics_data.json`)
   - All statistics in JSON format for further processing

## 🎯 Analytics Included

### 📊 Message Statistics
- Total messages and participants
- Messages per participant
- Word count analysis
- Character count distribution
- Media message count

### ⏰ Time Analysis
- Hourly activity patterns
- Daily activity (by day of week)
- Monthly trends
- Most active periods
- Response time analysis

### 💬 Content Analysis
- Top 20 most used words
- Top 10 emojis
- Sentiment analysis
- Message length statistics

### 📱 Chat Insights
- Media vs text message ratio
- Deleted message count
- Activity heatmaps
- Conversation flow analysis

## 🔧 Supported Chat Formats

The tool automatically detects and parses various WhatsApp export formats:

```
# Format Examples:
12/25/21, 10:30 - John: Hello there!
[25/12/21, 22:30:45] Jane: How are you?
25.12.21, 14:20 - Mike: Great to see you
2021-12-25 10:30:00 - Sarah: Happy holidays!
12/25/21, 2:30 PM - Alex: Good afternoon
```

## 📈 Sample Output

The generated report includes:
- 📊 Interactive dashboard with 12+ visualizations
- 👥 Participant statistics and rankings
- 🕐 Time-based activity analysis
- 💭 Sentiment analysis results
- 🎯 Key insights and patterns
- ☁️ Beautiful word cloud
- 📱 Mobile-responsive design

## 🛠️ Advanced Usage

### Custom Analysis
```python
from whatsapp_analyzer import WhatsAppAnalyzer

analyzer = WhatsAppAnalyzer('path/to/chat.txt')
analyzer.parse_messages()
analytics = analyzer.generate_analytics()

# Access specific data
print(f"Total messages: {analytics['total_messages']}")
print(f"Most active participant: {max(analytics['message_stats'], key=lambda x: analytics['message_stats'][x]['message']['count'])}")
```

### Batch Processing
```bash
# Process multiple chat files
for file in *.txt; do
    python whatsapp_analyzer.py "$file" --output "analysis_$(basename "$file" .txt)"
done
```

## 🔒 Privacy & Security

- **All processing is done locally** - your chat data never leaves your computer
- No data is sent to any external servers
- Generated reports can be shared at your discretion
- Original chat files remain unchanged

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.

---

**Made with ❤️ for WhatsApp chat analysis**

*Enjoy exploring your chat patterns and insights!* 📊✨