import anthropic
import pandas as pd
import time
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
import logging
import os
from datetime import datetime

# Configuration
MAX_TITLES = 10  # Default limit for number of titles to process

class TibetanTranslator:
    def __init__(self, api_key_file: str, batch_size: int = 10):
        self.api_key = self.read_api_key(api_key_file)
        self.client = anthropic.Client(api_key=self.api_key)
        self.batch_size = batch_size
        self.setup_logging()
        
    @staticmethod
    def read_api_key(api_key_file: str) -> str:
        """Read API key from file"""
        try:
            with open(api_key_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file {api_key_file} not found")
        except Exception as e:
            raise Exception(f"Error reading API key: {str(e)}")

    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.basicConfig(
            filename=f'translation_log_{timestamp}.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def create_prompt(self, title_data: List[Dict]) -> str:
        """Create a detailed prompt for translation and analysis"""
        titles = [item['title'] for item in title_data]
        return f"""For each of the following Tibetan titles, please provide:
        1. Spelling correction of the Tibetan text if needed
        2. Linguistic analysis including:
           - Syntactic structure (identify the main noun phrases, verb phrases, and their relationships)
           - Key lexical components (identify important terms and technical vocabulary)
           - Any grammatical particles and their function
        3. English translation based on this analysis
        
        Please format the response as a JSON object with the following structure for each title:
        {{
            "original": "original Tibetan text",
            "corrected": "corrected Tibetan text (if different from original)",
            "analysis": {{
                "syntax": "syntactic analysis",
                "lexical": "lexical analysis",
                "particles": "particle analysis"
            }},
            "translation": "English translation"
        }}
        
        Tibetan titles to analyze:
        {json.dumps(titles, ensure_ascii=False, indent=2)}"""

    def process_batch(self, title_data: List[Dict]) -> List[Dict]:
        """Process a batch of titles and return detailed analysis"""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0,
                system="You are a highly skilled Tibetan language expert with deep knowledge of Tibetan grammar, orthography, and translation. You excel at detailed linguistic analysis and accurate translation.",
                messages=[
                    {
                        "role": "user",
                        "content": self.create_prompt(title_data)
                    }
                ]
            )
            
            # Extract JSON from response
            response_text = message.content[0].text
            try:
                analyses = json.loads(response_text)
                # Add book IDs to the analyses
                for analysis, data in zip(analyses, title_data):
                    analysis['book_id'] = data['id']
                return analyses
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON from response: {response_text}")
                return [{'book_id': data['id']} for data in title_data]
                
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            return [{'book_id': data['id']} for data in title_data]

    def translate_titles(self, input_file: str, output_base: str, limit: int = MAX_TITLES):
        """Translate titles from input CSV and save multiple output files"""
        # Read input file
        df = pd.read_csv(input_file)
        
        # Apply limit
        df = df.head(limit)
        
        # Prepare data with IDs
        title_data = [
            {'id': row['book_id'], 'title': row['tibetan_title']} 
            for _, row in df.iterrows()
        ]
        
        all_analyses = []
        processed = 0
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(title_data), self.batch_size)):
            batch = title_data[i:i + self.batch_size]
            
            # Process batch
            analyses = self.process_batch(batch)
            all_analyses.extend(analyses)
            
            processed += len(batch)
            logging.info(f"Processed {processed}/{len(title_data)} titles")
            
            # Save intermediate results
            if processed % 100 == 0:
                self.save_results(all_analyses, output_base, processed)
            
            # Rate limiting
            time.sleep(1)  # Adjust based on API limits
        
        # Save final results
        self.save_results(all_analyses, output_base)
        logging.info("Processing completed")
        
    def save_results(self, analyses: List[Dict], output_base: str, checkpoint: int = None):
        """Save results to multiple CSV files"""
        suffix = f"_partial_{checkpoint}" if checkpoint else ""
        
        # Prepare DataFrames for different aspects
        translations_data = []
        corrections_data = []
        analysis_data = []
        
        for analysis in analyses:
            if not analysis:  # Skip empty entries
                continue
                
            book_id = analysis.get('book_id', '')
                
            # Translations and corrections
            translations_data.append({
                'Book_ID': book_id,
                'Original': analysis.get('original', ''),
                'Translation': analysis.get('translation', '')
            })
            
            if analysis.get('original') != analysis.get('corrected'):
                corrections_data.append({
                    'Book_ID': book_id,
                    'Original': analysis.get('original', ''),
                    'Corrected': analysis.get('corrected', '')
                })
            
            # Linguistic analysis
            analysis_data.append({
                'Book_ID': book_id,
                'Original': analysis.get('original', ''),
                'Syntactic_Analysis': analysis.get('analysis', {}).get('syntax', ''),
                'Lexical_Analysis': analysis.get('analysis', {}).get('lexical', ''),
                'Particle_Analysis': analysis.get('analysis', {}).get('particles', '')
            })
        
        # Save to separate files
        pd.DataFrame(translations_data).to_csv(
            f"{output_base}_translations{suffix}.csv", 
            index=False, 
            encoding='utf-8'
        )
        
        if corrections_data:
            pd.DataFrame(corrections_data).to_csv(
                f"{output_base}_corrections{suffix}.csv", 
                index=False, 
                encoding='utf-8'
            )
            
        pd.DataFrame(analysis_data).to_csv(
            f"{output_base}_analysis{suffix}.csv", 
            index=False, 
            encoding='utf-8'
        )

# Example usage
if __name__ == "__main__":
    try:
        # Initialize translator with API key from file
        translator = TibetanTranslator(api_key_file="ankey.txt", batch_size=10)
        
        # Process titles with default limit of 1000
        translator.translate_titles(
            input_file="tibetan_titles.csv",
            output_base="tibetan_processed"
        )
        
        # Or specify a different limit
        # translator.translate_titles(
        #     input_file="tibetan_titles.csv",
        #     output_base="tibetan_processed",
        #     limit=100  # Process only 100 titles
        # )
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")
