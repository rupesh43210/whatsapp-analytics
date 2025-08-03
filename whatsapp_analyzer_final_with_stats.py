#!/usr/bin/env python3
"""
Complete WhatsApp Analytics with Comprehensive Statistics
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
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class CompleteWhatsAppAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.messages = []
        self.df = None
        self.participants = set()
        self.clean_names = {}
        
    def clean_participant_name(self, name):
        """Clean and standardize participant names"""
        if name in self.clean_names:
            return self.clean_names[name]
        
        # Remove common suffixes and prefixes
        cleaned = re.sub(r'@.*?(?=\s|$)', '', name)  # Remove @mentions
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Remove extra spaces
        
        # Extract first name and apartment/unit if available
        parts = cleaned.split()
        if len(parts) > 1:
            # Try to identify apartment numbers or units
            for part in parts:
                if re.match(r'[A-Z]-?\d+|[A-Z]\d+|\d+[A-Z]?', part):
                    first_name = parts[0]
                    unit = part
                    cleaned = f"{first_name} ({unit})"
                    break
            else:
                # Just use first and last name
                if len(parts) >= 2:
                    cleaned = f"{parts[0]} {parts[-1]}"
                else:
                    cleaned = parts[0]
        
        self.clean_names[name] = cleaned
        return cleaned
    
    def parse_messages(self):
        """Enhanced parser for various WhatsApp export formats"""
        patterns = [
            # Format: [DD/MM/YY, HH:MM:SS AM/PM] Name: Message
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}:\d{2})\s+(AM|PM)\]\s*([^:]+?):\s*(.*)',
            # Format: [DD/MM/YY, HH:MM:SS] Name: Message
            r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}:\d{2})\]\s*([^:]+?):\s*(.*)',
            # Format: DD/MM/YY, HH:MM - Name: Message
            r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2})\s*[-–]\s*([^:]+?):\s*(.*)',
        ]
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        current_message = None
        
        for line_num, line in enumerate(lines, 1):
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
                    elif len(groups) == 5:
                        if groups[2] in ['AM', 'PM']:
                            date_str, time_str, am_pm, sender, message = groups
                            time_str = f"{time_str} {am_pm}"
                        else:
                            date_str, time_str, sender, am_pm, message = groups
                            time_str = f"{time_str} {am_pm}"
                    
                    # Skip system messages
                    if message.strip().startswith('‎') or 'added' in message or 'created this group' in message:
                        continue
                    
                    clean_sender = self.clean_participant_name(sender.strip())
                    
                    current_message = {
                        'line_number': line_num,
                        'date_str': date_str,
                        'time_str': time_str,
                        'sender': clean_sender,
                        'original_sender': sender.strip(),
                        'message': message.strip()
                    }
                    self.participants.add(clean_sender)
                    message_found = True
                    break
            
            if not message_found and current_message:
                current_message['message'] += ' ' + line
        
        if current_message:
            self.messages.append(current_message)
        
        self._process_timestamps()
        print(f"📊 Analyzing {len(self.messages)} messages from {len(self.participants)} participants")
        self._create_dataframe()
        
    def _process_timestamps(self):
        """Process various timestamp formats with better handling"""
        for msg in self.messages:
            date_str = msg['date_str']
            time_str = msg['time_str']
            
            date_formats = [
                '%d/%m/%y', '%d/%m/%Y', '%m/%d/%y', '%m/%d/%Y',
                '%d.%m.%y', '%d.%m.%Y', '%Y-%m-%d'
            ]
            
            time_formats = [
                '%H:%M:%S %p', '%I:%M:%S %p', '%H:%M:%S', '%H:%M', '%I:%M %p'
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
                try:
                    timestamp = datetime.strptime(f"{date_str} {time_str[:8]}", '%d/%m/%y %H:%M:%S')
                except ValueError:
                    timestamp = datetime.now()
            
            msg['timestamp'] = timestamp
            msg['date'] = timestamp.date()
            msg['time'] = timestamp.time()
            msg['hour'] = timestamp.hour
            msg['day_of_week'] = timestamp.strftime('%A')
            msg['month'] = timestamp.strftime('%B')
            msg['year'] = timestamp.year
            msg['week_of_year'] = timestamp.isocalendar()[1]
    
    def _create_dataframe(self):
        """Create enhanced pandas DataFrame with additional features"""
        if not self.messages:
            raise ValueError("No messages found. Please check the file format.")
        
        self.df = pd.DataFrame(self.messages)
        self.df['word_count'] = self.df['message'].apply(lambda x: len(x.split()))
        self.df['char_count'] = self.df['message'].apply(len)
        self.df['emoji_count'] = self.df['message'].apply(lambda x: len([c for c in x if c in emoji.EMOJI_DATA]))
        self.df['is_media'] = self.df['message'].str.contains(r'<Media omitted>|<attached:|image omitted|video omitted|audio omitted|document omitted', case=False, na=False)
        self.df['is_deleted'] = self.df['message'].str.contains(r'This message was deleted|You deleted this message', case=False, na=False)
        self.df['is_question'] = self.df['message'].str.contains(r'\?', na=False)
        self.df['has_url'] = self.df['message'].str.contains(r'http[s]?://|www\.', case=False, na=False)
        
        # Add time-based features
        self.df['date_formatted'] = self.df['timestamp'].dt.strftime('%B %d, %Y')
        self.df['time_period'] = pd.cut(self.df['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night (12-6AM)', 'Morning (6-12PM)', 'Afternoon (12-6PM)', 'Evening (6-12AM)'],
                                       include_lowest=True)
    
    def create_simple_visualizations(self):
        """Create simple, working visualizations"""
        
        # Single chart approach for debugging
        participant_counts = self.df['sender'].value_counts().head(10)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(participant_counts.index),
            y=list(participant_counts.values),
            marker_color='#3498db',
            text=[f"{v:,}" for v in participant_counts.values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Messages by Participant",
            xaxis_title="Participants", 
            yaxis_title="Number of Messages",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def generate_comprehensive_statistics(self):
        """Generate comprehensive statistics"""
        daily_counts = self.df.groupby(self.df['timestamp'].dt.date).size()
        hourly_counts = self.df['hour'].value_counts()
        participant_counts = self.df['sender'].value_counts()
        
        # Participant detailed stats
        participant_stats = []
        for participant in participant_counts.index[:10]:
            participant_data = self.df[self.df['sender'] == participant]
            stats = {
                'name': participant,
                'messages': len(participant_data),
                'words': participant_data['word_count'].sum(),
                'avg_words': participant_data['word_count'].mean(),
                'characters': participant_data['char_count'].sum(),
                'media_files': participant_data['is_media'].sum(),
                'questions': participant_data['is_question'].sum(),
                'first_message': participant_data['timestamp'].min().strftime('%B %d, %Y'),
                'last_message': participant_data['timestamp'].max().strftime('%B %d, %Y')
            }
            participant_stats.append(stats)
        
        # Time-based analysis
        day_stats = []
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days_order:
            day_data = self.df[self.df['day_of_week'] == day]
            day_stats.append({
                'day': day,
                'messages': len(day_data),
                'percentage': (len(day_data) / len(self.df)) * 100,
                'avg_words': day_data['word_count'].mean() if len(day_data) > 0 else 0
            })
        
        # Hour breakdown
        hour_stats = []
        for hour in range(24):
            hour_data = self.df[self.df['hour'] == hour]
            hour_label = f"{hour:02d}:00"
            if hour == 0:
                hour_label = "12:00 AM"
            elif hour < 12:
                hour_label = f"{hour}:00 AM"
            elif hour == 12:
                hour_label = "12:00 PM"
            else:
                hour_label = f"{hour-12}:00 PM"
                
            hour_stats.append({
                'hour': hour_label,
                'messages': len(hour_data),
                'percentage': (len(hour_data) / len(self.df)) * 100 if len(hour_data) > 0 else 0
            })
        
        return {
            'total_messages': len(self.df),
            'total_participants': len(self.participants),
            'date_range': {
                'start': self.df['timestamp'].min().strftime('%B %d, %Y'),
                'end': self.df['timestamp'].max().strftime('%B %d, %Y'),
                'duration': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            },
            'busiest_date': {
                'date': daily_counts.idxmax().strftime('%B %d, %Y'),
                'count': daily_counts.max()
            },
            'peak_hour': {
                'hour': hourly_counts.idxmax(),
                'count': hourly_counts.max(),
                'formatted': f"{hourly_counts.idxmax()}:00"
            },
            'top_participant': {
                'name': participant_counts.index[0],
                'count': participant_counts.iloc[0]
            },
            'avg_daily': daily_counts.mean(),
            'participant_details': participant_stats,
            'day_breakdown': day_stats,
            'hour_breakdown': hour_stats,
            'content_stats': {
                'text_messages': len(self.df[~self.df['is_media']]),
                'media_files': self.df['is_media'].sum(),
                'questions': self.df['is_question'].sum(),
                'urls': self.df['has_url'].sum(),
                'total_words': self.df['word_count'].sum(),
                'total_characters': self.df['char_count'].sum(),
                'avg_message_length': self.df['word_count'].mean()
            }
        }
    
    def generate_complete_report(self, output_dir='complete_analysis'):
        """Generate complete report with comprehensive statistics"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.parse_messages()
        
        # Create simple working visualization first
        main_fig = self.create_simple_visualizations()
        stats = self.generate_comprehensive_statistics()
        
        # Generate participant table HTML
        participant_table = ""
        for i, p in enumerate(stats['participant_details']):
            participant_table += f"""
            <tr>
                <td>{i+1}</td>
                <td><strong>{p['name']}</strong></td>
                <td>{p['messages']:,}</td>
                <td>{p['words']:,}</td>
                <td>{p['avg_words']:.1f}</td>
                <td>{p['media_files']:,}</td>
                <td>{p['questions']:,}</td>
                <td>{p['first_message']}</td>
            </tr>
            """
        
        # Generate day breakdown table
        day_table = ""
        for day_stat in stats['day_breakdown']:
            day_table += f"""
            <tr>
                <td><strong>{day_stat['day']}</strong></td>
                <td>{day_stat['messages']:,}</td>
                <td>{day_stat['percentage']:.1f}%</td>
                <td>{day_stat['avg_words']:.1f}</td>
            </tr>
            """
        
        # Generate hour breakdown table (top 12 hours)
        hour_data = sorted(stats['hour_breakdown'], key=lambda x: x['messages'], reverse=True)[:12]
        hour_table = ""
        for hour_stat in hour_data:
            if hour_stat['messages'] > 0:
                hour_table += f"""
                <tr>
                    <td><strong>{hour_stat['hour']}</strong></td>
                    <td>{hour_stat['messages']:,}</td>
                    <td>{hour_stat['percentage']:.1f}%</td>
                </tr>
                """
        
        # Generate HTML with comprehensive statistics
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Complete WhatsApp Analytics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Arial', sans-serif; background: #f5f5f5; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
                .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 30px; background: #f8f9fa; }}
                .stat {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .stat-number {{ font-size: 2rem; font-weight: bold; color: #667eea; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
                .section {{ padding: 30px; }}
                .chart {{ background: white; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h2 {{ color: #333; margin-bottom: 20px; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
                h3 {{ color: #555; margin: 20px 0 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                th {{ background: #667eea; color: white; padding: 12px; text-align: left; font-weight: bold; }}
                td {{ padding: 12px; border-bottom: 1px solid #eee; }}
                tr:hover {{ background: #f8f9fa; }}
                .insights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .insight {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; }}
                .insight-title {{ font-weight: bold; color: #333; }}
                .insight-value {{ color: #28a745; font-size: 1.3rem; font-weight: bold; }}
                .content-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .content-stats {{ background: #e8f4fd; padding: 20px; border-radius: 10px; }}
                .content-stat {{ display: flex; justify-content: space-between; margin: 10px 0; }}
                .footer {{ background: #333; color: white; text-align: center; padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📱 Complete WhatsApp Analytics</h1>
                    <p>Comprehensive Group Chat Analysis</p>
                    <p>{stats['date_range']['start']} - {stats['date_range']['end']}</p>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div class="stat-number">{stats['total_messages']:,}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['total_participants']}</div>
                        <div class="stat-label">Participants</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['date_range']['duration']}</div>
                        <div class="stat-label">Days Active</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['avg_daily']:.0f}</div>
                        <div class="stat-label">Avg Daily Messages</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['content_stats']['total_words']:,}</div>
                        <div class="stat-label">Total Words</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">{stats['content_stats']['avg_message_length']:.1f}</div>
                        <div class="stat-label">Avg Words per Message</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Message Distribution</h2>
                    <div class="chart">
                        <div id="main-chart" style="width:100%;height:500px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Key Insights</h2>
                    <div class="insights">
                        <div class="insight">
                            <div class="insight-title">🔥 Busiest Day</div>
                            <div class="insight-value">{stats['busiest_date']['date']}</div>
                            <div>{stats['busiest_date']['count']:,} messages</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">⏰ Peak Hour</div>
                            <div class="insight-value">{stats['peak_hour']['formatted']}</div>
                            <div>{stats['peak_hour']['count']:,} messages</div>
                        </div>
                        <div class="insight">
                            <div class="insight-title">👑 Most Active</div>
                            <div class="insight-value">{stats['top_participant']['name']}</div>
                            <div>{stats['top_participant']['count']:,} messages</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="content-grid">
                        <div>
                            <h3>📝 Content Statistics</h3>
                            <div class="content-stats">
                                <div class="content-stat">
                                    <span>Text Messages:</span>
                                    <strong>{stats['content_stats']['text_messages']:,}</strong>
                                </div>
                                <div class="content-stat">
                                    <span>Media Files:</span>
                                    <strong>{stats['content_stats']['media_files']:,}</strong>
                                </div>
                                <div class="content-stat">
                                    <span>Questions Asked:</span>
                                    <strong>{stats['content_stats']['questions']:,}</strong>
                                </div>
                                <div class="content-stat">
                                    <span>URLs Shared:</span>
                                    <strong>{stats['content_stats']['urls']:,}</strong>
                                </div>
                                <div class="content-stat">
                                    <span>Total Characters:</span>
                                    <strong>{stats['content_stats']['total_characters']:,}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>👥 Participant Details</h2>
                    <p>Top 10 most active participants with detailed statistics</p>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Name</th>
                                <th>Messages</th>
                                <th>Total Words</th>
                                <th>Avg Words</th>
                                <th>Media Files</th>
                                <th>Questions</th>
                                <th>First Message</th>
                            </tr>
                        </thead>
                        <tbody>
                            {participant_table}
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>📅 Activity by Day of Week</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Day</th>
                                <th>Messages</th>
                                <th>Percentage</th>
                                <th>Avg Words</th>
                            </tr>
                        </thead>
                        <tbody>
                            {day_table}
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>⏰ Top Active Hours</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Hour</th>
                                <th>Messages</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            {hour_table}
                        </tbody>
                    </table>
                </div>
                
                <div class="footer">
                    <p>📊 Complete WhatsApp Analytics Report</p>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
            </div>
            
            <script>
                var chartData = {main_fig.to_json()};
                Plotly.newPlot('main-chart', chartData.data, chartData.layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """
        
        report_path = os.path.join(output_dir, 'complete_analytics_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path, stats

def main():
    parser = argparse.ArgumentParser(description='Complete WhatsApp Analytics Tool')
    parser.add_argument('file_path', help='Path to WhatsApp chat export file')
    parser.add_argument('--output', '-o', default='complete_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File '{args.file_path}' not found.")
        return
    
    print("🚀 Starting Complete WhatsApp Analytics...")
    print(f"📁 Input file: {args.file_path}")
    print(f"📁 Output directory: {args.output}")
    
    try:
        analyzer = CompleteWhatsAppAnalyzer(args.file_path)
        report_path, stats = analyzer.generate_complete_report(args.output)
        
        print(f"\n✅ Complete Analysis Done!")
        print(f"📊 Analyzed {stats['total_messages']:,} messages")
        print(f"👥 From {stats['total_participants']} participants")
        print(f"🔥 Busiest day: {stats['busiest_date']['date']} ({stats['busiest_date']['count']:,} messages)")
        print(f"⏰ Peak hour: {stats['peak_hour']['formatted']} ({stats['peak_hour']['count']:,} messages)")
        print(f"📝 Complete report: {report_path}")
        print(f"\n🌐 Open {report_path} to view the complete analytics!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()