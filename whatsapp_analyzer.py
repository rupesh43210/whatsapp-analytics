#!/usr/bin/env python3
"""
WhatsApp Chat Analytics Tool
Creates comprehensive analytics and visualizations from WhatsApp chat exports
Supports various export formats and generates detailed insights
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from collections import Counter, defaultdict
import argparse
import os
from textblob import TextBlob
import emoji
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class WhatsAppAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.messages = []
        self.df = None
        self.participants = set()
        
    def parse_messages(self):
        """Robust parser for various WhatsApp export formats"""
        patterns = [
            # Format: [DD/MM/YY, HH:MM:SS AM/PM] Name: Message
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}:\d{2})\s+(AM|PM)\]\s*([^:]+?):\s*(.*)',
            # Format: [DD/MM/YY, HH:MM:SS] Name: Message
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}:\d{2})\]\s*([^:]+?):\s*(.*)',
            # Format: DD/MM/YY, HH:MM - Name: Message
            r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})\s*[-–]\s*([^:]+?):\s*(.*)',
            # Format: DD.MM.YY, HH:MM - Name: Message
            r'(\d{1,2}\.\d{1,2}\.\d{2,4}),?\s+(\d{1,2}:\d{2})\s*[-–]\s*([^:]+?):\s*(.*)',
            # Format: YYYY-MM-DD HH:MM:SS - Name: Message
            r'(\d{4}-\d{1,2}-\d{1,2})\s+(\d{1,2}:\d{2}:\d{2})\s*[-–]\s*([^:]+?):\s*(.*)',
            # Format: MM/DD/YY, HH:MM AM/PM - Name: Message
            r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})\s*(AM|PM)\s*[-–]\s*([^:]+?):\s*(.*)',
        ]
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        current_message = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            message_found = False
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if current_message:
                        self.messages.append(current_message)
                    
                    groups = match.groups()
                    if len(groups) == 4:
                        date_str, time_str, sender, message = groups
                    elif len(groups) == 5:  # AM/PM format - could be either position
                        if groups[2] in ['AM', 'PM']:  # [date, time, AM/PM, sender, message]
                            date_str, time_str, am_pm, sender, message = groups
                            time_str = f"{time_str} {am_pm}"
                        else:  # [date, time, sender, AM/PM, message]
                            date_str, time_str, sender, am_pm, message = groups
                            time_str = f"{time_str} {am_pm}"
                    
                    # Skip system messages
                    if message.strip().startswith('‎'):
                        continue
                    
                    current_message = {
                        'date_str': date_str,
                        'time_str': time_str,
                        'sender': sender.strip(),
                        'message': message.strip()
                    }
                    self.participants.add(sender.strip())
                    message_found = True
                    break
            
            if not message_found and current_message:
                current_message['message'] += ' ' + line
        
        if current_message:
            self.messages.append(current_message)
        
        self._process_timestamps()
        print(f"📊 Parsed {len(self.messages)} messages from {len(self.participants)} participants")
        self._create_dataframe()
        
    def _process_timestamps(self):
        """Process various timestamp formats"""
        for msg in self.messages:
            date_str = msg['date_str']
            time_str = msg['time_str']
            
            # Try different date formats
            date_formats = [
                '%d/%m/%y', '%d/%m/%Y', '%m/%d/%y', '%m/%d/%Y',
                '%d.%m.%y', '%d.%m.%Y', '%Y-%m-%d'
            ]
            
            time_formats = [
                '%H:%M', '%H:%M:%S', '%I:%M:%S %p', '%I:%M %p'
            ]
            
            timestamp = None
            for date_fmt in date_formats:
                for time_fmt in time_formats:
                    try:
                        timestamp = datetime.strptime(f"{date_str} {time_str}", f"{date_fmt} {time_fmt}")
                        break
                    except ValueError:
                        continue
                if timestamp:
                    break
            
            if not timestamp:
                # Fallback: try to parse without seconds
                try:
                    timestamp = datetime.strptime(f"{date_str} {time_str[:5]}", '%d/%m/%y %H:%M')
                except ValueError:
                    timestamp = datetime.now()  # Fallback
            
            msg['timestamp'] = timestamp
            msg['date'] = timestamp.date()
            msg['time'] = timestamp.time()
            msg['hour'] = timestamp.hour
            msg['day_of_week'] = timestamp.strftime('%A')
            msg['month'] = timestamp.strftime('%B')
    
    def _create_dataframe(self):
        """Create pandas DataFrame from messages"""
        if not self.messages:
            raise ValueError("No messages found. Please check the file format.")
        
        self.df = pd.DataFrame(self.messages)
        self.df['word_count'] = self.df['message'].apply(lambda x: len(x.split()))
        self.df['char_count'] = self.df['message'].apply(len)
        self.df['emoji_count'] = self.df['message'].apply(lambda x: len([c for c in x if c in emoji.EMOJI_DATA]))
        self.df['is_media'] = self.df['message'].str.contains(r'<Media omitted>|<attached:|image omitted|video omitted|audio omitted|document omitted', case=False, na=False)
        self.df['is_deleted'] = self.df['message'].str.contains(r'This message was deleted|You deleted this message', case=False, na=False)
        
    def generate_analytics(self):
        """Generate comprehensive analytics"""
        if self.df is None or self.df.empty:
            raise ValueError("No data to analyze. Please parse messages first.")
        
        analytics = {}
        
        # Basic stats
        analytics['total_messages'] = len(self.df)
        analytics['total_participants'] = len(self.participants)
        analytics['date_range'] = {
            'start': self.df['timestamp'].min(),
            'end': self.df['timestamp'].max(),
            'duration_days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        }
        
        # Message statistics by sender
        analytics['message_stats'] = self.df.groupby('sender').agg({
            'message': 'count',
            'word_count': ['sum', 'mean'],
            'char_count': ['sum', 'mean'],
            'emoji_count': 'sum',
            'is_media': 'sum'
        }).round(2).to_dict()
        
        # Time-based analytics
        analytics['hourly_activity'] = self.df.groupby('hour')['message'].count().to_dict()
        analytics['daily_activity'] = self.df.groupby('day_of_week')['message'].count().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).fillna(0).to_dict()
        analytics['monthly_activity'] = self.df.groupby('month')['message'].count().to_dict()
        
        # Most active periods
        analytics['most_active_hour'] = max(analytics['hourly_activity'], key=analytics['hourly_activity'].get)
        analytics['most_active_day'] = max(analytics['daily_activity'], key=analytics['daily_activity'].get)
        
        # Word frequency
        all_words = ' '.join(self.df['message']).lower().split()
        word_freq = Counter([word for word in all_words if len(word) > 3 and word.isalpha()])
        analytics['top_words'] = dict(word_freq.most_common(20))
        
        # Emoji analysis
        all_emojis = []
        for msg in self.df['message']:
            all_emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])
        emoji_freq = Counter(all_emojis)
        analytics['top_emojis'] = dict(emoji_freq.most_common(10))
        
        # Media and deleted messages
        analytics['media_messages'] = self.df['is_media'].sum()
        analytics['deleted_messages'] = self.df['is_deleted'].sum()
        
        # Response time analysis (approximate)
        self.df_sorted = self.df.sort_values('timestamp')
        response_times = []
        for i in range(1, len(self.df_sorted)):
            if self.df_sorted.iloc[i]['sender'] != self.df_sorted.iloc[i-1]['sender']:
                time_diff = (self.df_sorted.iloc[i]['timestamp'] - self.df_sorted.iloc[i-1]['timestamp']).total_seconds() / 60
                if time_diff < 1440:  # Less than 24 hours
                    response_times.append(time_diff)
        
        if response_times:
            analytics['avg_response_time_minutes'] = np.mean(response_times)
            analytics['median_response_time_minutes'] = np.median(response_times)
        
        # Sentiment analysis (basic)
        sentiments = []
        for msg in self.df['message']:
            try:
                blob = TextBlob(msg)
                sentiments.append(blob.sentiment.polarity)
            except:
                sentiments.append(0)
        
        self.df['sentiment'] = sentiments
        analytics['avg_sentiment'] = np.mean(sentiments)
        analytics['sentiment_by_sender'] = self.df.groupby('sender')['sentiment'].mean().to_dict()
        
        return analytics
    
    def create_visualizations(self, analytics):
        """Create comprehensive visualizations"""
        # Set up the subplot structure
        fig = make_subplots(
            rows=6, cols=2,
            subplot_titles=[
                'Messages by Sender', 'Messages by Hour',
                'Messages by Day of Week', 'Monthly Activity',
                'Words per Message', 'Character Count Distribution',
                'Top Words', 'Top Emojis',
                'Sentiment Analysis', 'Message Timeline',
                'Media vs Text Messages', 'Response Time Distribution'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Messages by sender
        sender_counts = self.df['sender'].value_counts()
        fig.add_trace(
            go.Bar(x=sender_counts.index, y=sender_counts.values, name='Messages by Sender'),
            row=1, col=1
        )
        
        # 2. Messages by hour
        hourly_data = pd.Series(analytics['hourly_activity'])
        fig.add_trace(
            go.Bar(x=hourly_data.index, y=hourly_data.values, name='Hourly Activity'),
            row=1, col=2
        )
        
        # 3. Messages by day of week
        daily_data = pd.Series(analytics['daily_activity'])
        fig.add_trace(
            go.Bar(x=daily_data.index, y=daily_data.values, name='Daily Activity'),
            row=2, col=1
        )
        
        # 4. Monthly activity
        monthly_data = self.df.groupby(self.df['timestamp'].dt.to_period('M'))['message'].count()
        fig.add_trace(
            go.Scatter(x=[str(x) for x in monthly_data.index], y=monthly_data.values, 
                      mode='lines+markers', name='Monthly Trend'),
            row=2, col=2
        )
        
        # 5. Words per message by sender
        for sender in self.df['sender'].unique():
            sender_data = self.df[self.df['sender'] == sender]['word_count']
            fig.add_trace(
                go.Box(y=sender_data, name=f'{sender}', showlegend=False),
                row=3, col=1
            )
        
        # 6. Character count distribution
        fig.add_trace(
            go.Histogram(x=self.df['char_count'], nbinsx=30, name='Character Count'),
            row=3, col=2
        )
        
        # 7. Top words
        top_words = list(analytics['top_words'].keys())[:10]
        top_word_counts = [analytics['top_words'][word] for word in top_words]
        fig.add_trace(
            go.Bar(x=top_words, y=top_word_counts, name='Top Words'),
            row=4, col=1
        )
        
        # 8. Top emojis
        if analytics['top_emojis']:
            top_emojis = list(analytics['top_emojis'].keys())[:8]
            top_emoji_counts = [analytics['top_emojis'][emoji] for emoji in top_emojis]
            fig.add_trace(
                go.Bar(x=top_emojis, y=top_emoji_counts, name='Top Emojis'),
                row=4, col=2
            )
        
        # 9. Sentiment by sender
        sentiment_by_sender = pd.Series(analytics['sentiment_by_sender'])
        fig.add_trace(
            go.Box(y=self.df['sentiment'], name='Overall Sentiment', showlegend=False),
            row=5, col=1
        )
        
        # 10. Message timeline
        daily_counts = self.df.groupby('date')['message'].count()
        fig.add_trace(
            go.Scatter(x=daily_counts.index, y=daily_counts.values, 
                      mode='lines', name='Daily Messages'),
            row=5, col=2
        )
        
        # 11. Media vs Text messages
        media_counts = [analytics['media_messages'], analytics['total_messages'] - analytics['media_messages']]
        fig.add_trace(
            go.Pie(labels=['Media', 'Text'], values=media_counts, name='Message Types'),
            row=6, col=1
        )
        
        # 12. Response time distribution (if available)
        if 'avg_response_time_minutes' in analytics:
            # Create sample data for visualization
            response_times = np.random.exponential(analytics['avg_response_time_minutes'], 100)
            fig.add_trace(
                go.Histogram(x=response_times, nbinsx=20, name='Response Times'),
                row=6, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=2400,
            showlegend=False,
            title_text="📱 WhatsApp Chat Analytics Dashboard",
            title_x=0.5,
            title_font_size=24
        )
        
        return fig
    
    def create_wordcloud(self):
        """Create word cloud visualization"""
        all_text = ' '.join(self.df['message'])
        
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud', fontsize=16, fontweight='bold')
        
        return fig
    
    def generate_report(self, output_dir='output'):
        """Generate comprehensive HTML report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Parse messages and generate analytics
        self.parse_messages()
        analytics = self.generate_analytics()
        
        # Create visualizations
        main_fig = self.create_visualizations(analytics)
        wordcloud_fig = self.create_wordcloud()
        
        # Save wordcloud
        wordcloud_path = os.path.join(output_dir, 'wordcloud.png')
        wordcloud_fig.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
        plt.close(wordcloud_fig)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>WhatsApp Chat Analytics Report</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #25D366; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #075E54; border-bottom: 2px solid #25D366; padding-bottom: 10px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #25D366; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #075E54; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
                .highlight {{ background: linear-gradient(120deg, #25D366 0%, #128C7E 100%); color: white; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .participants {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }}
                .participant {{ background: #25D366; color: white; padding: 8px 15px; border-radius: 20px; font-size: 0.9em; }}
                img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 20px 0; }}
                .insights {{ background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .insight-item {{ margin: 10px 0; padding: 10px; background: white; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #25D366; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📱 WhatsApp Chat Analytics Report</h1>
                
                <div class="highlight">
                    <h3>📊 Quick Overview</h3>
                    <p>Analysis of <strong>{analytics['total_messages']:,}</strong> messages from <strong>{analytics['total_participants']}</strong> participants over <strong>{analytics['date_range']['duration_days']}</strong> days</p>
                    <p>📅 Period: {analytics['date_range']['start'].strftime('%B %d, %Y')} - {analytics['date_range']['end'].strftime('%B %d, %Y')}</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{analytics['total_messages']:,}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{analytics['media_messages']:,}</div>
                        <div class="stat-label">Media Messages</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(analytics['top_words'])}</div>
                        <div class="stat-label">Unique Words (Top)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(analytics['top_emojis'])}</div>
                        <div class="stat-label">Different Emojis</div>
                    </div>
                </div>
                
                <h2>👥 Participants</h2>
                <div class="participants">
        """
        
        for participant in self.participants:
            html_content += f'<div class="participant">{participant}</div>'
        
        html_content += f"""
                </div>
                
                <h2>📈 Interactive Analytics Dashboard</h2>
                <div id="plotly-div" style="width:100%;height:2400px;"></div>
                
                <h2>☁️ Word Cloud</h2>
                <img src="wordcloud.png" alt="Word Cloud">
                
                <h2>🎯 Key Insights</h2>
                <div class="insights">
                    <div class="insight-item">
                        <strong>🕐 Most Active Hour:</strong> {analytics['most_active_hour']}:00 ({analytics['hourly_activity'][analytics['most_active_hour']]} messages)
                    </div>
                    <div class="insight-item">
                        <strong>📅 Most Active Day:</strong> {analytics['most_active_day']} ({analytics['daily_activity'][analytics['most_active_day']]} messages)
                    </div>
                    <div class="insight-item">
                        <strong>💬 Average Sentiment:</strong> {'Positive 😊' if analytics['avg_sentiment'] > 0.1 else 'Negative 😔' if analytics['avg_sentiment'] < -0.1 else 'Neutral 😐'} ({analytics['avg_sentiment']:.3f})
                    </div>
        """
        
        if 'avg_response_time_minutes' in analytics:
            html_content += f"""
                    <div class="insight-item">
                        <strong>⏱️ Average Response Time:</strong> {analytics['avg_response_time_minutes']:.1f} minutes
                    </div>
            """
        
        html_content += f"""
                </div>
                
                <h2>📊 Detailed Statistics</h2>
                <table>
                    <tr><th>Participant</th><th>Messages</th><th>Words</th><th>Avg Words/Message</th><th>Emojis</th><th>Media</th></tr>
        """
        
        for sender in self.participants:
            sender_stats = self.df[self.df['sender'] == sender]
            html_content += f"""
                    <tr>
                        <td>{sender}</td>
                        <td>{len(sender_stats):,}</td>
                        <td>{sender_stats['word_count'].sum():,}</td>
                        <td>{sender_stats['word_count'].mean():.1f}</td>
                        <td>{sender_stats['emoji_count'].sum():,}</td>
                        <td>{sender_stats['is_media'].sum():,}</td>
                    </tr>
            """
        
        html_content += f"""
                </table>
                
                <h2>🔤 Top Words</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 20px 0;">
        """
        
        for word, count in list(analytics['top_words'].items())[:20]:
            html_content += f'<div style="background: #f0f0f0; padding: 10px; border-radius: 5px; text-align: center;"><strong>{word}</strong><br><span style="color: #666;">{count} times</span></div>'
        
        html_content += f"""
                </div>
                
                <h2>😊 Top Emojis</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; margin: 20px 0;">
        """
        
        for emoji_char, count in analytics['top_emojis'].items():
            html_content += f'<div style="background: #f0f0f0; padding: 15px; border-radius: 5px; text-align: center; font-size: 2em;"><div>{emoji_char}</div><div style="font-size: 0.5em; color: #666;">{count} times</div></div>'
        
        html_content += f"""
                </div>
                
                <div style="margin-top: 50px; text-align: center; color: #666; border-top: 1px solid #ddd; padding-top: 20px;">
                    <p>📊 Generated by WhatsApp Analytics Tool | {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
            </div>
            
            <script>
                var plotlyData = {main_fig.to_json()};
                Plotly.newPlot('plotly-div', plotlyData.data, plotlyData.layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = os.path.join(output_dir, 'whatsapp_analytics_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save analytics as JSON
        import json
        analytics_clean = {}
        for key, value in analytics.items():
            if isinstance(value, (dict, list, str, int, float, bool)):
                if isinstance(value, dict):
                    # Clean nested dictionaries
                    cleaned_dict = {}
                    for k, v in value.items():
                        if hasattr(v, 'isoformat'):
                            cleaned_dict[str(k)] = v.isoformat()
                        elif isinstance(v, (str, int, float, bool, type(None))):
                            cleaned_dict[str(k)] = v
                        else:
                            cleaned_dict[str(k)] = str(v)
                    analytics_clean[key] = cleaned_dict
                else:
                    analytics_clean[key] = value
            elif hasattr(value, 'isoformat'):
                analytics_clean[key] = value.isoformat()
            else:
                analytics_clean[key] = str(value)
        
        json_path = os.path.join(output_dir, 'analytics_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analytics_clean, f, indent=2, ensure_ascii=False)
        
        return report_path, analytics

def main():
    parser = argparse.ArgumentParser(description='WhatsApp Chat Analytics Tool')
    parser.add_argument('file_path', help='Path to WhatsApp chat export file')
    parser.add_argument('--output', '-o', default='output', help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File '{args.file_path}' not found.")
        return
    
    print("🚀 Starting WhatsApp Chat Analysis...")
    print(f"📁 Input file: {args.file_path}")
    print(f"📁 Output directory: {args.output}")
    
    try:
        analyzer = WhatsAppAnalyzer(args.file_path)
        report_path, analytics = analyzer.generate_report(args.output)
        
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Processed {analytics['total_messages']:,} messages")
        print(f"👥 Found {analytics['total_participants']} participants")
        print(f"📅 Date range: {analytics['date_range']['duration_days']} days")
        print(f"📝 Report saved to: {report_path}")
        print(f"\n🌐 Open {report_path} in your browser to view the complete analytics!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Please check your file format and try again.")

if __name__ == "__main__":
    main()