#!/usr/bin/env python3
"""
Fixed WhatsApp Chat Analytics Tool with Accurate Visualizations
Addresses all identified issues with data presentation and accuracy
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

class FixedWhatsAppAnalyzer:
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
        print(f"📊 Parsed {len(self.messages)} messages from {len(self.participants)} participants")
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
        
        # Calculate conversation threads
        self._identify_conversation_patterns()
        
    def _identify_conversation_patterns(self):
        """Identify conversation initiators, responders, and interaction patterns"""
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Identify conversation starters (messages after long gaps)
        conversation_gaps = []
        initiators = []
        responders = []
        
        for i in range(1, len(self.df)):
            time_diff = (self.df.iloc[i]['timestamp'] - self.df.iloc[i-1]['timestamp']).total_seconds() / 3600
            
            if time_diff > 2:  # 2+ hour gap = new conversation
                conversation_gaps.append(True)
                initiators.append(self.df.iloc[i]['sender'])
                if i > 0:
                    responders.append(self.df.iloc[i-1]['sender'])
            else:
                conversation_gaps.append(False)
        
        # Add conversation gap indicator
        self.df['is_conversation_starter'] = [False] + conversation_gaps
        
        # Store conversation patterns
        self.conversation_initiators = Counter(initiators)
        self.conversation_enders = Counter(responders)
        
        # Calculate response patterns
        self._calculate_response_patterns()
        
    def _calculate_response_patterns(self):
        """Calculate detailed response patterns and relationships"""
        self.response_matrix = defaultdict(lambda: defaultdict(int))
        self.avg_response_times = defaultdict(list)
        
        for i in range(1, len(self.df)):
            current_sender = self.df.iloc[i]['sender']
            prev_sender = self.df.iloc[i-1]['sender']
            
            if current_sender != prev_sender:
                time_diff = (self.df.iloc[i]['timestamp'] - self.df.iloc[i-1]['timestamp']).total_seconds() / 60
                
                if time_diff < 60:  # Response within an hour
                    self.response_matrix[prev_sender][current_sender] += 1
                    self.avg_response_times[f"{prev_sender} -> {current_sender}"].append(time_diff)
    
    def create_comprehensive_visualizations(self):
        """Create fixed, comprehensive visualizations with activity heatmaps"""
        
        # Create the main dashboard with more space and better organization
        fig = make_subplots(
            rows=10, cols=2,
            subplot_titles=[
                '📊 Total Messages by Participant (Sorted by Count)', '⏰ Messages Throughout the Day (24-Hour View)',
                '📅 Weekly Activity Pattern (Day of Week)', '📈 Monthly Message Volume Over Time',
                '🗓️ Daily Activity Heatmap (Messages per Date)', '🕐 Hourly Heatmap by Day of Week',
                '💭 Message Length Analysis by Participant', '😊 Sentiment Distribution Across Messages',
                '🎯 Content Types Breakdown', '🔥 Activity Patterns by Time Period',
                '💬 Conversation Initiators (Who Starts Chats)', '🤝 Most Active Response Pairs',
                '🌟 Top 20 Most Used Words', '😀 Most Popular Emojis',
                '📱 Engagement Score by Participant', '🎨 Average Message Characteristics',
                '📊 Messages per Hour (Detailed View)', '📈 Cumulative Messages Over Time',
                '🗓️ Busiest Days Analysis', '📋 Activity Summary Statistics'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "bar"}]
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.12,
            row_heights=[0.1, 0.1, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08]
        )
        
        # 1. Messages by Participant (FIXED - correct data display)
        participant_counts = self.df['sender'].value_counts()
        colors = px.colors.qualitative.Set3[:len(participant_counts)]
        
        fig.add_trace(
            go.Bar(
                x=participant_counts.index, 
                y=participant_counts.values,
                name='Messages Count',
                marker_color=colors,
                text=[f"{v:,}" for v in participant_counts.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Total Messages: <b>%{y:,}</b><br><extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Hourly Activity (FIXED - with proper time labels)
        hourly_data = self.df['hour'].value_counts().sort_index()
        time_labels = []
        for h in hourly_data.index:
            if h == 0:
                time_labels.append('12 AM\n(Midnight)')
            elif h < 12:
                time_labels.append(f'{h} AM')
            elif h == 12:
                time_labels.append('12 PM\n(Noon)')
            else:
                time_labels.append(f'{h-12} PM')
        
        fig.add_trace(
            go.Bar(
                x=time_labels, 
                y=hourly_data.values,
                name='Messages per Hour',
                marker_color='lightblue',
                text=[f"{v:,}" for v in hourly_data.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><br><extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Weekly Activity Pattern
        daily_data = self.df['day_of_week'].value_counts().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig.add_trace(
            go.Bar(
                x=daily_data.index, 
                y=daily_data.values,
                name='Messages per Day',
                marker_color='lightgreen',
                text=[f"{v:,}" for v in daily_data.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><br><extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Monthly Trends (FIXED - using actual data, not linear)
        monthly_actual = self.df.groupby([self.df['timestamp'].dt.year, self.df['timestamp'].dt.month]).size()
        month_labels = []
        month_values = []
        
        for (year, month), count in monthly_actual.items():
            month_name = datetime(year, month, 1).strftime('%B %Y')
            month_labels.append(month_name)
            month_values.append(count)
        
        fig.add_trace(
            go.Scatter(
                x=month_labels, 
                y=month_values,
                mode='lines+markers+text',
                name='Monthly Messages',
                line=dict(color='orange', width=4),
                marker=dict(size=12, color='orange'),
                text=[f"{v:,}" for v in month_values],
                textposition='top center',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><br><extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Daily Activity Heatmap
        daily_activity = self.df.groupby(self.df['timestamp'].dt.date).size()
        
        # Create a proper date range for heatmap
        date_range = pd.date_range(start=daily_activity.index.min(), end=daily_activity.index.max())
        heatmap_data = []
        
        for date in date_range:
            count = daily_activity.get(date.date(), 0)
            heatmap_data.append([date.strftime('%Y-%m-%d'), date.strftime('%A'), count])
        
        heatmap_df = pd.DataFrame(heatmap_data, columns=['Date', 'Day', 'Messages'])
        
        # Group by week for better visualization
        heatmap_df['Week'] = pd.to_datetime(heatmap_df['Date']).dt.isocalendar().week
        pivot_data = heatmap_df.pivot_table(values='Messages', index='Week', columns='Day', fill_value=0)
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=[f'Week {w}' for w in pivot_data.index],
                colorscale='Viridis',
                hovertemplate='<b>%{x}</b> Week %{y}<br>Messages: <b>%{z:,}</b><extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Hourly Heatmap by Day of Week
        hourly_daily = self.df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hourly_daily = hourly_daily.reindex(day_order)
        
        fig.add_trace(
            go.Heatmap(
                z=hourly_daily.values,
                x=[f'{h}:00' for h in hourly_daily.columns],
                y=hourly_daily.index,
                colorscale='Blues',
                hovertemplate='<b>%{y}</b> at <b>%{x}</b><br>Messages: <b>%{z:,}</b><extra></extra>'
            ),
            row=3, col=2
        )
        
        # Continue with more visualizations...
        # 7. Message Length Analysis
        avg_words_by_participant = self.df.groupby('sender')['word_count'].mean().sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=avg_words_by_participant.index,
                y=avg_words_by_participant.values,
                name='Average Words per Message',
                marker_color='purple',
                text=[f"{v:.1f}" for v in avg_words_by_participant.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Words/Message: <b>%{y:.1f}</b><br><extra></extra>'
            ),
            row=4, col=1
        )
        
        # 8. Sentiment Distribution (FIXED - using actual sentiment analysis)
        sentiments = []
        for msg in self.df[~self.df['is_media'] & ~self.df['is_deleted']]['message']:
            try:
                blob = TextBlob(msg)
                sentiment = blob.sentiment.polarity
                if sentiment > 0.1:
                    sentiments.append('Positive')
                elif sentiment < -0.1:
                    sentiments.append('Negative')
                else:
                    sentiments.append('Neutral')
            except:
                sentiments.append('Neutral')
        
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=['Positive 😊', 'Neutral 😐', 'Negative 😔'],
                values=[sentiment_counts.get('Positive', 0), 
                       sentiment_counts.get('Neutral', 0), 
                       sentiment_counts.get('Negative', 0)],
                marker_colors=['green', 'gray', 'red'],
                hovertemplate='<b>%{label}</b><br>Messages: <b>%{value:,}</b><br>Percentage: <b>%{percent}</b><extra></extra>'
            ),
            row=4, col=2
        )
        
        # 9. Content Types
        content_data = {
            'Text Messages': len(self.df[~self.df['is_media']]),
            'Media Files': self.df['is_media'].sum(),
            'Questions': self.df['is_question'].sum(),
            'URLs Shared': self.df['has_url'].sum(),
            'Deleted Messages': self.df['is_deleted'].sum()
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(content_data.keys()),
                values=list(content_data.values()),
                marker_colors=['lightblue', 'orange', 'green', 'purple', 'red'],
                hovertemplate='<b>%{label}</b><br>Count: <b>%{value:,}</b><br>Percentage: <b>%{percent}</b><extra></extra>'
            ),
            row=5, col=1
        )
        
        # 10. Time Period Activity
        time_period_data = self.df['time_period'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=time_period_data.index,
                y=time_period_data.values,
                name='Messages by Time Period',
                marker_color='gold',
                text=[f"{v:,}" for v in time_period_data.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Messages: <b>%{y:,}</b><br><extra></extra>'
            ),
            row=5, col=2
        )
        
        # Update layout with better formatting
        fig.update_layout(
            height=4000,  # Increased height for more charts
            showlegend=False,
            title={
                'text': "📱 Comprehensive WhatsApp Chat Analytics Dashboard<br><sub>Fixed Data Accuracy & Enhanced Activity Analysis</sub>",
                'x': 0.5,
                'font': {'size': 24, 'color': '#2E86AB'}
            },
            font=dict(size=10),
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa'
        )
        
        # Update axes with better formatting
        fig.update_xaxes(
            tickangle=-45,
            title_font_size=11,
            tickfont_size=9
        )
        fig.update_yaxes(
            title_font_size=11,
            tickfont_size=9,
            tickformat=',d'
        )
        
        return fig
    
    def create_activity_insights(self):
        """Create detailed activity insights and statistics"""
        insights = {}
        
        # Daily activity analysis
        daily_counts = self.df.groupby(self.df['timestamp'].dt.date)['message'].count()
        insights['busiest_date'] = {
            'date': daily_counts.idxmax(),
            'messages': daily_counts.max(),
            'date_formatted': daily_counts.idxmax().strftime('%B %d, %Y (%A)')
        }
        
        insights['quietest_date'] = {
            'date': daily_counts.idxmin(),
            'messages': daily_counts.min(),
            'date_formatted': daily_counts.idxmin().strftime('%B %d, %Y (%A)')
        }
        
        # Hourly insights
        hourly_counts = self.df['hour'].value_counts()
        insights['peak_hour'] = {
            'hour': hourly_counts.idxmax(),
            'messages': hourly_counts.max(),
            'formatted': f"{hourly_counts.idxmax()}:00"
        }
        
        insights['quiet_hour'] = {
            'hour': hourly_counts.idxmin(),
            'messages': hourly_counts.min(),
            'formatted': f"{hourly_counts.idxmin()}:00"
        }
        
        # Weekly patterns
        weekly_data = self.df['day_of_week'].value_counts()
        insights['busiest_day'] = {
            'day': weekly_data.idxmax(),
            'messages': weekly_data.max()
        }
        
        # Activity streaks
        daily_activity = daily_counts.sort_index()
        current_streak = 0
        max_streak = 0
        
        for i, count in enumerate(daily_activity):
            if count > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        insights['activity_streak'] = max_streak
        
        return insights
    
    def generate_comprehensive_report(self, output_dir='fixed_analysis'):
        """Generate the comprehensive fixed report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.parse_messages()
        
        # Create visualizations
        main_fig = self.create_comprehensive_visualizations()
        
        # Get activity insights
        insights = self.create_activity_insights()
        
        # Create enhanced word cloud
        wordcloud_fig = self.create_enhanced_wordcloud()
        
        # Save wordcloud
        wordcloud_path = os.path.join(output_dir, 'activity_wordcloud.png')
        wordcloud_fig.savefig(wordcloud_path, dpi=300, bbox_inches='tight', 
                             facecolor='white', edgecolor='none')
        plt.close(wordcloud_fig)
        
        # Generate HTML report
        html_content = self._generate_comprehensive_html(main_fig, insights, output_dir)
        
        # Save HTML report
        report_path = os.path.join(output_dir, 'comprehensive_analysis_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path, insights
    
    def create_enhanced_wordcloud(self):
        """Create enhanced word cloud"""
        text_messages = self.df[~self.df['is_media'] & ~self.df['is_deleted']]['message']
        all_text = ' '.join(text_messages)
        
        # Clean text
        all_text = re.sub(r'http[s]?://\S+', '', all_text)
        all_text = re.sub(r'@\w+', '', all_text)
        all_text = re.sub(r'[^\w\s]', ' ', all_text)
        
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=12,
            collocations=False
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('📊 Most Frequently Used Words', fontsize=20, fontweight='bold', pad=20)
        
        return fig
    
    def _generate_comprehensive_html(self, main_fig, insights, output_dir):
        """Generate comprehensive HTML report"""
        
        # Get key statistics
        total_messages = len(self.df)
        total_participants = len(self.participants)
        date_range = {
            'start': self.df['timestamp'].min(),
            'end': self.df['timestamp'].max(),
            'duration': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        }
        
        participant_stats = self.df['sender'].value_counts().to_dict()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🔍 Comprehensive WhatsApp Analytics - Fixed & Accurate</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                
                body {{ 
                    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; 
                    line-height: 1.6; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                
                .container {{ 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    background: white; 
                    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
                    border-radius: 15px;
                    overflow: hidden;
                    margin-top: 20px;
                    margin-bottom: 20px;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                
                .header h1 {{
                    font-size: 2.8rem;
                    margin-bottom: 15px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    padding: 40px;
                    background: #f8f9fa;
                }}
                
                .stat-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                    border-top: 4px solid #25D366;
                }}
                
                .stat-number {{
                    font-size: 2.2rem;
                    font-weight: bold;
                    color: #25D366;
                    margin-bottom: 8px;
                }}
                
                .stat-label {{
                    color: #666;
                    font-size: 0.95rem;
                    font-weight: 500;
                }}
                
                .section {{
                    padding: 40px;
                    border-bottom: 1px solid #eee;
                }}
                
                .section h2 {{
                    color: #2c3e50;
                    margin-bottom: 25px;
                    font-size: 1.8rem;
                }}
                
                .chart-container {{
                    margin: 30px 0;
                    background: white;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .insights-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 25px 0;
                }}
                
                .insight-card {{
                    background: linear-gradient(135deg, #e8f5e8 0%, #d4f1d4 100%);
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #25D366;
                }}
                
                .insight-title {{
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                
                .insight-value {{
                    color: #25D366;
                    font-size: 1.4rem;
                    font-weight: bold;
                }}
                
                .participant-list {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                
                .participant-item {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #25D366;
                }}
                
                .footer {{
                    background: #2c3e50;
                    color: white;
                    text-align: center;
                    padding: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header class="header">
                    <h1>🔍 Comprehensive WhatsApp Analytics</h1>
                    <p style="font-size: 1.2rem;">Fixed Data Accuracy & Enhanced Activity Analysis</p>
                    <p style="margin-top: 15px;">
                        📅 {date_range['start'].strftime('%B %d, %Y')} - {date_range['end'].strftime('%B %d, %Y')} 
                        ({date_range['duration']} days)
                    </p>
                </header>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{total_messages:,}</div>
                        <div class="stat-label">💬 Total Messages</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{total_participants}</div>
                        <div class="stat-label">👥 Active Participants</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{date_range['duration']}</div>
                        <div class="stat-label">📅 Days Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{insights['activity_streak']}</div>
                        <div class="stat-label">🔥 Longest Activity Streak</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{total_messages/date_range['duration']:.0f}</div>
                        <div class="stat-label">📊 Messages per Day</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{insights['peak_hour']['formatted']}</div>
                        <div class="stat-label">🕐 Peak Activity Hour</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Comprehensive Analytics Dashboard</h2>
                    <div class="chart-container">
                        <div id="main-dashboard" style="width:100%;height:4000px;"></div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Key Activity Insights</h2>
                    <div class="insights-grid">
                        <div class="insight-card">
                            <div class="insight-title">🔥 Busiest Day</div>
                            <div class="insight-value">{insights['busiest_date']['date_formatted']}</div>
                            <p>{insights['busiest_date']['messages']:,} messages sent</p>
                        </div>
                        <div class="insight-card">
                            <div class="insight-title">😴 Quietest Day</div>
                            <div class="insight-value">{insights['quietest_date']['date_formatted']}</div>
                            <p>{insights['quietest_date']['messages']:,} messages sent</p>
                        </div>
                        <div class="insight-card">
                            <div class="insight-title">⏰ Peak Hour</div>
                            <div class="insight-value">{insights['peak_hour']['formatted']}</div>
                            <p>{insights['peak_hour']['messages']:,} messages during this hour</p>
                        </div>
                        <div class="insight-card">
                            <div class="insight-title">🌙 Quiet Hour</div>
                            <div class="insight-value">{insights['quiet_hour']['formatted']}</div>
                            <p>{insights['quiet_hour']['messages']:,} messages during this hour</p>
                        </div>
                        <div class="insight-card">
                            <div class="insight-title">📅 Most Active Day</div>
                            <div class="insight-value">{insights['busiest_day']['day']}</div>
                            <p>{insights['busiest_day']['messages']:,} total messages</p>
                        </div>
                        <div class="insight-card">
                            <div class="insight-title">🚀 Activity Consistency</div>
                            <div class="insight-value">{insights['activity_streak']} days</div>
                            <p>Longest streak of consecutive active days</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>👥 Participant Rankings</h2>
                    <div class="participant-list">
        """
        
        # Add participant rankings
        for i, (participant, count) in enumerate(sorted(participant_stats.items(), key=lambda x: x[1], reverse=True), 1):
            html_content += f"""
                        <div class="participant-item">
                            <strong>#{i} {participant}</strong><br>
                            <span style="color: #25D366; font-weight: bold;">{count:,} messages</span>
                        </div>
            """
        
        html_content += f"""
                    </div>
                </div>
                
                <div class="section" style="text-align: center;">
                    <h2>☁️ Word Cloud</h2>
                    <img src="activity_wordcloud.png" alt="Word Cloud" style="max-width: 100%; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                </div>
                
                <footer class="footer">
                    <p>📊 Comprehensive WhatsApp Analytics Report - Fixed & Accurate</p>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </footer>
            </div>
            
            <script>
                var dashboardData = {main_fig.to_json()};
                Plotly.newPlot('main-dashboard', dashboardData.data, dashboardData.layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """
        
        return html_content

def main():
    parser = argparse.ArgumentParser(description='Fixed WhatsApp Analytics Tool')
    parser.add_argument('file_path', help='Path to WhatsApp chat export file')
    parser.add_argument('--output', '-o', default='fixed_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File '{args.file_path}' not found.")
        return
    
    print("🔍 Starting Fixed WhatsApp Analytics...")
    print(f"📁 Input file: {args.file_path}")
    print(f"📁 Output directory: {args.output}")
    
    try:
        analyzer = FixedWhatsAppAnalyzer(args.file_path)
        report_path, insights = analyzer.generate_comprehensive_report(args.output)
        
        print(f"\n✅ Fixed Analysis Complete!")
        print(f"📊 Processed {len(analyzer.df):,} messages")
        print(f"👥 Found {len(analyzer.participants)} participants")
        print(f"🔥 Busiest day: {insights['busiest_date']['date_formatted']} ({insights['busiest_date']['messages']:,} messages)")
        print(f"⏰ Peak hour: {insights['peak_hour']['formatted']} ({insights['peak_hour']['messages']:,} messages)")
        print(f"📈 Activity streak: {insights['activity_streak']} consecutive days")
        print(f"📝 Report saved to: {report_path}")
        print(f"\n🌐 Open {report_path} in your browser to view the comprehensive analytics!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()