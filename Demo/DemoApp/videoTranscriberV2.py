import os
import numpy as np
from moviepy import VideoFileClip
from pyannote.audio import Pipeline
from pyannote.audio import Model
import whisper
from textblob import TextBlob
import json
import soundfile as sf
import time
from pyannote.audio import Inference
from scipy.spatial.distance import cdist
import torch
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LiveMeetingAnalyzer:
    def __init__(self, video_path, buffer_size=12, overlap=0):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.step_size = buffer_size - overlap

        self.audio_path = "temp_audio.wav"
        self.current_position = 0
        self.total_duration = None
        self.stop_processing = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token="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"
        ).to(self.device)
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token="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"
        ).to(self.device)
        self.inference = Inference(self.embedding_model, window="whole").to(self.device)
        self.transcription_model = whisper.load_model("tiny",device="cuda")

        # Speaker tracking
        self.speaker_embeddings = {}  # {speaker_id: embedding_vector}
        self.embedding_threshold = 0.77  # Similarity threshold for speaker matching
        self.accumulated_results = {
            'meeting_duration': 0,
            'speaker_statistics': {},
            'transcript': []
        }
        self.vectorizer = None
        self.overall_model = None
        self.overall_topics = []
        self.accumulated_texts = []
        self.speaker_accumulated_texts = {}

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

    def extract_audio(self):
        """Extract full audio from video file"""
        print("Extracting audio from video...")
        video = VideoFileClip(self.video_path)
        self.total_duration = video.duration
        video.audio.write_audiofile(self.audio_path, fps=16000)
        return self.audio_path

    def get_audio_segment(self, start_time, end_time):
        """Extract audio segment for the current buffer"""
        audio, sr = sf.read(self.audio_path, start=int(start_time * 16000),
                            frames=int((end_time - start_time) * 16000))
        return audio, sr

    def process_buffer(self, start_time, end_time):
        """Process a single buffer of audio with speaker embedding tracking"""
        try:
            audio_segment, sr = self.get_audio_segment(start_time, end_time)

            if len(audio_segment.shape) > 1:
                audio_segment = audio_segment.mean(axis=1)
            audio_segment = audio_segment.astype(np.float32)

            temp_buffer_path = "temp_buffer.wav"
            sf.write(temp_buffer_path, audio_segment, sr)

            diarization = self.diarization_pipeline(
                temp_buffer_path, min_speakers=1, max_speakers=5
            )

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                abs_start = start_time + turn.start
                abs_end = start_time + turn.end
                speaker_id = self.match_speaker_embedding(temp_buffer_path, turn)

                # Only add segments if the speaker has meaningful transcription
                if speaker_id != "SPEAKER_UNKNOWN":  # Exclude "ghost" speakers
                    segments.append({
                        'start': abs_start,
                        'end': abs_end,
                        'speaker': speaker_id,
                        'duration': turn.end - turn.start
                    })

            transcription = self.transcription_model.transcribe(
                audio_segment, language="english", verbose=False, fp16=False
            )

            # Filter out the ghost speakers (those with no transcription text)
            buffer_results = self.process_transcription(segments, transcription, start_time)

            # Only keep results where transcription text is meaningful (not empty or just whitespace)
            buffer_results = [result for result in buffer_results if result['text'].strip()]

            self.update_results(buffer_results, segments)
            self.save_results()

            if os.path.exists(temp_buffer_path):
                os.remove(temp_buffer_path)

            return True
        except Exception as e:
            print(f"Error processing buffer: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def match_speaker_embedding(self, audio_path, turn):
        """Generate embeddings for the speaker and match with existing embeddings."""
        try:
            # Extract audio segment for the current turn
            start = int(turn.start * 16000)
            end = int(turn.end * 16000)
            audio, sr = sf.read(audio_path, start=start, frames=end - start)

            # Check if the audio is too short to process
            if len(audio) < 16000:  # Minimum length for processing (1 second of audio at 16kHz)
                print(f"Skipping segment {turn.start}s to {turn.end}s due to short audio.")
                return "SPEAKER_UNKNOWN"

            # Save temporary audio file for the segment
            temp_audio_path = "temp_segment.wav"
            sf.write(temp_audio_path, audio, sr)

            # Generate speaker embedding using pyannote's Inference
            embedding = self.inference(temp_audio_path)  # Shape: (512,)
            print(f"Generated embedding shape: {embedding.shape}")

            # Reshape embedding to 2D
            embedding = embedding.reshape(1, -1)  # Now shape will be (1, 512)
            print(f"Reshaped embedding for cdist: {embedding.shape}")

            # Compare with existing speaker embeddings
            best_match = None
            best_similarity = float("inf")
            for speaker_id, ref_embedding in self.speaker_embeddings.items():
                # Compute cosine distance
                similarity = cdist(embedding, ref_embedding, metric="cosine")[0, 0]
                if similarity < best_similarity and similarity < self.embedding_threshold:
                    best_match = speaker_id
                    best_similarity = similarity

            # If no match is found, create a new speaker ID
            if not best_match:
                best_match = f"SPEAKER_{len(self.speaker_embeddings):02d}"
                self.speaker_embeddings[best_match] = embedding

            return best_match
        except Exception as e:
            print(f"Error generating speaker embedding: {e}")
            return f"SPEAKER_UNKNOWN"
    def process_transcription(self, segments, transcription, buffer_start):
        """Process transcription for current buffer"""
        results = []

        for segment in transcription['segments']:
            # Adjust timestamps relative to video start
            start_time = buffer_start + segment['start']
            end_time = buffer_start + segment['end']
            text = segment['text']

            # Find matching speaker segment
            speaker = None
            for s in segments:
                if (start_time >= s['start'] and start_time < s['end']) or \
                        (end_time > s['start'] and end_time <= s['end']):
                    speaker = s['speaker']
                    break

            if speaker and text.strip():
                sentiment = self.analyze_sentiment(text)
                results.append({
                    'start': start_time,
                    'end': end_time,
                    'speaker': speaker,
                    'text': text,
                    'sentiment': sentiment
                })

        return results

    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def get_custom_stop_words(self):
        """Get extended stop words list including common filler words"""
        custom_stops = [
            'like', 'just', 'um', 'uh', 'well', 'sort', 'kind', 'yeah', 'yes', 'know',
            'mean', 'right', 'going', 'get', 'got', 'gonna', 'would', 'could', 'should',
            'really', 'say', 'saying', 'said', 'way', 'thing', 'things', 'think', 'thinking',
            'actually', 'basically', 'certainly', 'definitely', 'probably', 'possibly',
            'ok', 'okay', 'hey', 'oh', 'ow', 'wow', 'ah', 'uhm', 'like', 'so'
        ]
        custom_stops.extend(stopwords.words('english'))
        return list(dict.fromkeys(custom_stops))

    def clean_word(self, word):
        """Clean and validate a word"""
        cleaned = ''.join(c for c in word.lower() if c.isalpha())
        if len(cleaned) < 3:
            return None
        return cleaned

    def generate_topic_name(self, top_words, word_weights):
        """Generate a descriptive name for a topic based on its top words and weights"""
        significant_words = []
        custom_stops = self.get_custom_stop_words()

        for word, weight in zip(top_words, word_weights):
            cleaned_word = self.clean_word(word)
            if cleaned_word and cleaned_word not in custom_stops:
                formatted_word = cleaned_word.capitalize()
                if len(significant_words) < 3 and formatted_word not in significant_words:
                    significant_words.append(formatted_word)

        if len(significant_words) < 2:
            return "General Discussion"

        if len(significant_words) == 2:
            return f"{significant_words[0]} & {significant_words[1]}"
        return f"{significant_words[0]}, {significant_words[1]} & {significant_words[2]}"

    def perform_topic_modeling(self, texts, update_overall=False, num_topics=5, num_words=15):
        """Perform topic modeling on transcribed texts with improved error handling"""
        # Ensure we have enough meaningful text to analyze
        filtered_texts = [text for text in texts if len(text.split()) >= 3]  # Only include texts with 3+ words

        if not filtered_texts or len(filtered_texts) < 3:  # Require at least 3 meaningful segments
            print(f"Not enough text data for topic modeling (found {len(filtered_texts)} valid segments)")
            return [], None, None

        try:
            # Initialize or update vectorizer with more restrictive parameters
            if self.vectorizer is None:
                self.vectorizer = CountVectorizer(
                    max_df=0.95,
                    min_df=1,
                    stop_words=self.get_custom_stop_words(),
                    token_pattern=r'\b[a-zA-Z]{3,}\b',
                    max_features=500,
                )
                doc_term_matrix = self.vectorizer.fit_transform(filtered_texts)
            else:
                doc_term_matrix = self.vectorizer.transform(filtered_texts)

            # Check if we have enough terms
            if doc_term_matrix.shape[1] < 3:  # Need at least 3 terms
                print(f"Not enough unique terms found (found {doc_term_matrix.shape[1]} terms)")
                return [], None, None

            # Adjust number of topics based on available terms
            adjusted_num_topics = min(num_topics, doc_term_matrix.shape[1] // 3, len(filtered_texts) // 3)
            if adjusted_num_topics < 1:
                adjusted_num_topics = 1

            # Create and fit LDA model with adjusted parameters
            lda_model = LatentDirichletAllocation(
                n_components=adjusted_num_topics,
                random_state=42,
                max_iter=10,
                learning_decay=0.5,
                learning_method='online',
                n_jobs=-1,
                doc_topic_prior=0.9,  # Added to handle sparse data better
                topic_word_prior=0.9  # Added to handle sparse data better
            )

            # Fit the model with error handling
            try:
                doc_topic_distributions = lda_model.fit_transform(doc_term_matrix)
            except Exception as e:
                print(f"Error during LDA fitting: {str(e)}")
                return [], None, None

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                # Avoid division by zero
                topic_sum = np.sum(topic)
                if topic_sum == 0:
                    continue

                # Normalize topic
                normalized_topic = topic / topic_sum

                top_words_idx = normalized_topic.argsort()[:-num_words - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                word_weights = [normalized_topic[i] for i in top_words_idx]

                topic_name = self.generate_topic_name(top_words, word_weights)

                # Calculate topic weight with error handling
                try:
                    topic_weight = float(np.mean(doc_topic_distributions[:, topic_idx]))
                except:
                    topic_weight = 0.0

                display_words = []
                for word in top_words:
                    cleaned = self.clean_word(word)
                    if cleaned and cleaned not in self.get_custom_stop_words():
                        display_words.append(word)
                    if len(display_words) >= 10:
                        break

                if display_words:  # Only add topic if it has meaningful words
                    topics.append({
                        'topic_id': topic_idx,
                        'name': topic_name,
                        'top_words': display_words,
                        'weight': topic_weight
                    })

            if update_overall and topics:  # Only update if we have valid topics
                self.overall_model = lda_model
                self.overall_topics = topics

            return topics, doc_topic_distributions, lda_model

        except Exception as e:
            print(f"Warning: Topic modeling error - {str(e)}")
            import traceback
            traceback.print_exc()
            return [], None, None

    def calculate_topic_similarity(self, speaker_model, speaker_topics):
        """Calculate similarity between speaker topics and overall meeting topics"""
        if not speaker_model or not self.overall_model:
            return []

        similarities = []
        speaker_topic_term = speaker_model.components_
        overall_topic_term = self.overall_model.components_

        similarity_matrix = cosine_similarity(speaker_topic_term, overall_topic_term)

        for i, speaker_topic in enumerate(speaker_topics):
            best_match_idx = np.argmax(similarity_matrix[i])
            similarity_score = float(similarity_matrix[i][best_match_idx])

            similarities.append({
                'speaker_topic': speaker_topic['name'],
                'overall_topic': self.overall_topics[best_match_idx]['name'],
                'similarity_score': similarity_score
            })

        return similarities

    def update_results(self, buffer_results, segments):
        """Update accumulated results with buffer results"""
        # Add texts to accumulated texts for topic modeling
        new_texts = [result['text'] for result in buffer_results if len(result['text'].strip()) > 10]
        if new_texts:
            self.accumulated_texts.extend(new_texts)

            # Accumulate texts per speaker
            for result in buffer_results:
                if len(result['text'].strip()) > 10:
                    speaker = result['speaker']
                    if speaker not in self.speaker_accumulated_texts:
                        self.speaker_accumulated_texts[speaker] = []
                    self.speaker_accumulated_texts[speaker].append(result['text'])

        # Perform topic modeling on accumulated texts
        if len(self.accumulated_texts) >= 10:
            overall_topics, _, _ = self.perform_topic_modeling(
                self.accumulated_texts,
                update_overall=True,
                num_topics=1
            )

            # Initialize topics in accumulated_results if not present
            if 'topics' not in self.accumulated_results:
                self.accumulated_results['topics'] = {
                    'overall': [],
                    'per_speaker': {}
                }

            # Update overall topics
            if overall_topics:
                self.accumulated_results['topics']['overall'] = overall_topics

            # Process each speaker's topics
            for speaker, texts in self.speaker_accumulated_texts.items():
                if texts:  # Process if we have any texts for this speaker
                    topics, _, speaker_model = self.perform_topic_modeling(texts, num_topics=1)

                    if topics and speaker_model:
                        topic_similarities = self.calculate_topic_similarity(speaker_model, topics)
                        # Update just this speaker's topics while preserving others
                        self.accumulated_results['topics']['per_speaker'][speaker] = {
                            'topics': topics,
                            'topic_similarities': topic_similarities
                        }

        # Update transcript
        self.accumulated_results['transcript'].extend(buffer_results)

        # Update speaker statistics
        speaker_durations = {}
        total_duration = 0

        # Calculate durations for current segments
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['duration']

            # Only accumulate stats for valid speakers (excluding ghost speakers)
            if speaker != "SPEAKER_UNKNOWN" and len(
                    [result for result in buffer_results if result['speaker'] == speaker]) > 0:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
                total_duration += duration

        # Update speaker statistics, but filter out those with insignificant speaking time
        for speaker, duration in speaker_durations.items():
            if speaker not in self.accumulated_results['speaker_statistics']:
                self.accumulated_results['speaker_statistics'][speaker] = {
                    'speaking_time': 0,
                    'sentiment': {'polarity_sum': 0, 'subjectivity_sum': 0, 'count': 0}
                }

            self.accumulated_results['speaker_statistics'][speaker]['speaking_time'] += duration

        # Update sentiment statistics
        for result in buffer_results:
            speaker = result['speaker']
            sentiment = result['sentiment']
            if speaker != "SPEAKER_UNKNOWN" and len(
                    [r for r in buffer_results if r['speaker'] == speaker and r['text'].strip()]) > 0:
                stats = self.accumulated_results['speaker_statistics'][speaker]
                stats['sentiment']['polarity_sum'] += sentiment['polarity']
                stats['sentiment']['subjectivity_sum'] += sentiment['subjectivity']
                stats['sentiment']['count'] += 1

        # Update meeting duration
        self.accumulated_results['meeting_duration'] = max(
            self.accumulated_results['meeting_duration'],
            max(segment['end'] for segment in segments)
        )
    def save_results(self):
        """Save current results to JSON file"""
        # Calculate percentages and averages
        total_time = sum(s['speaking_time'] for s in self.accumulated_results['speaker_statistics'].values())

        output_results = {
            'meeting_duration': self.accumulated_results['meeting_duration'],
            'speaker_statistics': {},
            'transcript': self.accumulated_results['transcript'],
            'topics': {
                'overall': self.overall_topics,  # Add overall topics
                'per_speaker': {}  # Will be populated below
            }
        }

        for speaker, stats in self.accumulated_results['speaker_statistics'].items():
            sentiment_count = stats['sentiment']['count']
            output_results['speaker_statistics'][speaker] = {
                'speaking_time_percentage': (stats['speaking_time'] / total_time * 100) if total_time > 0 else 0,
                'sentiment': {
                    'average_polarity': stats['sentiment'][
                                            'polarity_sum'] / sentiment_count if sentiment_count > 0 else 0,
                    'average_subjectivity': stats['sentiment'][
                                                'subjectivity_sum'] / sentiment_count if sentiment_count > 0 else 0
                }
            }

            # Add speaker-specific topic information if available
            if 'topics' in self.accumulated_results and 'per_speaker' in self.accumulated_results['topics']:
                speaker_topics = self.accumulated_results['topics']['per_speaker'].get(speaker, {})
                if speaker_topics:
                    output_results['topics']['per_speaker'][speaker] = {
                        'topics': speaker_topics.get('topics', []),
                        'topic_similarities': speaker_topics.get('topic_similarities', [])
                    }

        # Save results to JSON file
        video_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   '_static', 'DemoApp', f'meeting_analysis_{video_filename}.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_results, f, indent=4, ensure_ascii=False)

    def analyze_meeting(self):
        """Perform live meeting analysis"""
        try:
            # Extract full audio first
            self.extract_audio()

            print(f"Starting live analysis of {self.total_duration:.2f} seconds of content...")

            # Process video in buffers
            while self.current_position < self.total_duration and not self.stop_processing:
                buffer_end = min(self.current_position + self.buffer_size, self.total_duration)
                print(f"\nProcessing segment {self.current_position:.1f}s to {buffer_end:.1f}s")

                # Process current buffer
                success = self.process_buffer(self.current_position, buffer_end)
                if not success:
                    print(f"Error processing buffer at {self.current_position:.1f}s")

                # Advance position by step size
                self.current_position += self.step_size

                # Simulate real-time processing
                time.sleep(self.step_size)  # Wait for step_size seconds

            print("\nAnalysis complete!")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            # Cleanup
            if os.path.exists(self.audio_path):
                os.remove(self.audio_path)


def main():
    video_path = input("Enter the path to your video file: ")
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return

    try:
        analyzer = LiveMeetingAnalyzer(video_path)
        analyzer.analyze_meeting()
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()