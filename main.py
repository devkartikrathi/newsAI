import os
import logging
import pandas as pd
from dateutil import parser
import isodate
from textblob import TextBlob

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    logging.error("Missing YOUTUBE_API_KEY environment variable!")
    raise ValueError("Set the YOUTUBE_API_KEY environment variable with your YouTube API key.")

YOUTUBE_CHANNEL_IDS = [
    "UCttspZesZIDEwwpVIgoZtWQ",
    "UCRWFSbif-RFENbBrSiez1DA",
]

def analyze_sentiment(text):
    """
    Perform simple sentiment analysis using TextBlob.
    Returns polarity as a float.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

class YouTubeAnalytics:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def get_channel_stats(self, channel_ids):
        """
        Fetch channel details (snippet, contentDetails, statistics) for a list of channel IDs.
        Returns a DataFrame with channel statistics.
        """
        all_data = []
        try:
            request = self.youtube.channels().list(
                part='snippet,contentDetails,statistics',
                id=",".join(channel_ids)
            )
            response = request.execute()
            for item in response.get('items', []):
                data = {
                    'channelId': item.get('id'),
                    'channelName': item['snippet'].get('title'),
                    'publishedAt': item['snippet'].get('publishedAt'),
                    'subscriberCount': int(item['statistics'].get('subscriberCount', 0)),
                    'viewCount': int(item['statistics'].get('viewCount', 0)),
                    'videoCount': int(item['statistics'].get('videoCount', 0)),
                    'uploadsPlaylistId': item['contentDetails']['relatedPlaylists'].get('uploads')
                }
                all_data.append(data)
        except HttpError as e:
            logging.error(f"YouTube API error while fetching channel stats: {e.resp.status} - {e.content}")
        return pd.DataFrame(all_data)
    
    def get_video_ids(self, playlist_id):
        """
        Retrieve all video IDs from a channel's uploads playlist (with pagination).
        """
        video_ids = []
        next_page_token = None
        while True:
            try:
                request = self.youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()
                for item in response.get('items', []):
                    vid = item['contentDetails'].get('videoId')
                    if vid:
                        video_ids.append(vid)
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            except HttpError as e:
                logging.error(f"Error fetching video IDs from playlist {playlist_id}: {e.resp.status} - {e.content}")
                break
        return video_ids
    
    def get_video_details(self, video_ids):
        """
        Retrieve video details (snippet, contentDetails, statistics) for a list of video IDs.
        Returns a DataFrame with video metadata.
        """
        all_video_info = []
        for i in range(0, len(video_ids), 50):
            try:
                request = self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(video_ids[i:i+50])
                )
                response = request.execute()
                for video in response.get('items', []):
                    stats_to_keep = {
                        'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                        'statistics': ['viewCount', 'likeCount', 'favoriteCount', 'commentCount'],
                        'contentDetails': ['duration', 'definition', 'caption']
                    }
                    video_info = {'video_id': video.get('id')}
                    for section, keys in stats_to_keep.items():
                        for key in keys:
                            video_info[key] = video.get(section, {}).get(key)
                    all_video_info.append(video_info)
            except HttpError as e:
                logging.error(f"Error fetching video details: {e.resp.status} - {e.content}")
        return pd.DataFrame(all_video_info)
    
    def get_live_streams(self, channel_id, max_results=5):
        """
        Check for currently active live streams on a YouTube channel.
        Returns a list of live video data.
        """
        live_videos = []
        try:
            request = self.youtube.search().list(
                part="snippet",
                channelId=channel_id,
                eventType="live",
                type="video",
                maxResults=max_results
            )
            response = request.execute()
            for item in response.get('items', []):
                live_video = {
                    "videoId": item["id"].get("videoId"),
                    "title": item["snippet"].get("title"),
                    "publishedAt": item["snippet"].get("publishedAt")
                }
                live_videos.append(live_video)
        except HttpError as e:
            logging.error(f"Error fetching live streams for channel {channel_id}: {e.resp.status} - {e.content}")
        return live_videos
    
    def preprocess_video_data(self, video_df):
        """
        Convert and enrich YouTube video data for analysis.
          - Convert count columns to numeric types.
          - Parse published dates and extract day names.
          - Convert ISO 8601 duration to seconds.
          - Count number of tags.
          - Compute like and comment ratios (per 1000 views).
          - Compute title length.
        """
        numeric_cols = ['viewCount', 'likeCount', 'favoriteCount', 'commentCount']
        for col in numeric_cols:
            video_df[col] = pd.to_numeric(video_df[col], errors='coerce').fillna(0).astype(int)
        
        video_df['publishedAt'] = pd.to_datetime(video_df['publishedAt'], errors='coerce')
        video_df['publishDayName'] = video_df['publishedAt'].dt.day_name()
        video_df['durationSecs'] = video_df['duration'].apply(
            lambda x: isodate.parse_duration(x).total_seconds() if pd.notnull(x) else None
        )
        video_df['tagsCount'] = video_df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        video_df['likeRatio'] = video_df.apply(
            lambda row: (row['likeCount'] / row['viewCount'] * 1000) if row['viewCount'] else 0, axis=1
        )
        video_df['commentRatio'] = video_df.apply(
            lambda row: (row['commentCount'] / row['viewCount'] * 1000) if row['viewCount'] else 0, axis=1
        )
        video_df['titleLength'] = video_df['title'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        return video_df
    
    def get_audience_insights(self, channel_id, start_date, end_date):
    
        if not hasattr(self, 'analytics_credentials') or self.analytics_credentials is None:
            raise ValueError("OAuth2 credentials for YouTube Analytics are required for audience insights.")
        
        from googleapiclient.discovery import build
        
        # Build the YouTube Analytics API client using OAuth2 credentials.
        youtube_analytics = build('youtubeAnalytics', 'v2', credentials=self.analytics_credentials)
        
        # Retrieve demographic data (age group and gender)
        demographics_response = youtube_analytics.reports().query(
            ids=f"channel=={channel_id}",
            startDate=start_date,
            endDate=end_date,
            metrics="viewerPercentage",
            dimensions="ageGroup,gender",
            sort="ageGroup,gender"
        ).execute()
        
        demo_headers = [header['name'] for header in demographics_response.get('columnHeaders', [])]
        demo_rows = demographics_response.get('rows', [])
        demographics_df = pd.DataFrame(demo_rows, columns=demo_headers) if demo_rows else pd.DataFrame()
        
        geography_response = youtube_analytics.reports().query(
            ids=f"channel=={channel_id}",
            startDate=start_date,
            endDate=end_date,
            metrics="viewerPercentage",
            dimensions="country",
            sort="country"
        ).execute()
        
        geo_headers = [header['name'] for header in geography_response.get('columnHeaders', [])]
        geo_rows = geography_response.get('rows', [])
        geography_df = pd.DataFrame(geo_rows, columns=geo_headers) if geo_rows else pd.DataFrame()
        
        return {
            "demographics": demographics_df,
            "geography": geography_df
        }

def main():
    yt_analytics = YouTubeAnalytics(YOUTUBE_API_KEY)
    
    # --- Collect Channel Statistics ---
    logging.info("Collecting YouTube channel statistics...")
    channel_stats = yt_analytics.get_channel_stats(YOUTUBE_CHANNEL_IDS)
    logging.info(f"Retrieved statistics for {len(channel_stats)} channels.")
    
    # --- Collect Video Data ---
    all_videos_df = pd.DataFrame()
    for _, row in channel_stats.iterrows():
        channel_name = row['channelName']
        playlist_id = row['uploadsPlaylistId']
        logging.info(f"Processing channel: {channel_name}")
        video_ids = yt_analytics.get_video_ids(playlist_id)
        logging.info(f"Found {len(video_ids)} videos for channel: {channel_name}")
        if video_ids:
            channel_videos = yt_analytics.get_video_details(video_ids)
            all_videos_df = pd.concat([all_videos_df, channel_videos], ignore_index=True)
    
    # --- Preprocess Video Data ---
    logging.info("Preprocessing video data...")
    all_videos_df = yt_analytics.preprocess_video_data(all_videos_df)
    
    # --- Example: Sentiment Analysis on Video Titles ---
    logging.info("Performing sentiment analysis on video titles...")
    all_videos_df['titleSentiment'] = all_videos_df['title'].apply(lambda x: analyze_sentiment(x))
    
    # --- (Optional) Check for Live Streams on each channel ---
    live_streams_data = {}
    for _, row in channel_stats.iterrows():
        channel_id = row['channelId']
        live_streams = yt_analytics.get_live_streams(channel_id)
        live_streams_data[row['channelName']] = live_streams
    logging.info("Live stream data collected.")
    
    # --- Save Data ---
    all_videos_df.to_csv("youtube_video_data.csv", index=False)
    pd.DataFrame(live_streams_data).to_csv("youtube_live_streams.csv", index=False)
    logging.info("YouTube data collection complete. CSV files saved.")
    logging.info("Sample video data:")
    logging.info(all_videos_df.head())

if __name__ == "__main__":
    main()
