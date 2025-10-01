#!/usr/bin/env python3
"""
Automated Research Paper Generation Pipeline

This script:
1. Reads research topic from my_research_topic.md
2. Uses OpenAI to extract and enhance search keywords
3. Finds relevant papers using Scholarly (Google Scholar) and Semantic Scholar API,
4. Generates Paper #1 with just the topic
5. Generates Paper #2 with topic + found references
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict
import re

# Import AI Researcher
from ai_researcher import CycleResearcher

# OpenAI for intelligent keyword extraction
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  OpenAI not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

# Scholarly for Google Scholar
try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    print("⚠️  Scholarly not installed. Run: pip install scholarly")
    SCHOLARLY_AVAILABLE = False


class AutomatedResearchPipeline:
    """Automated pipeline for research paper generation with intelligent reference finding"""
    
    def __init__(self, openai_api_key=None):
        """Initialize the pipeline"""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            print("⚠️  OpenAI API key not provided. Will use basic keyword extraction.")
        
        self.researcher = None
        
    def read_research_topic(self, filepath: str) -> Dict[str, str]:
        """
        Read and parse the research topic markdown file
        
        Args:
            filepath: Path to the markdown file
            
        Returns:
            Dictionary with parsed sections
        """
        print(f"📖 Reading research topic from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse markdown sections
        topic_data = {
            'raw_content': content,
            'title': '',
            'research_question': '',
            'background': '',
            'approach': '',
            'keywords': ''
        }
        
        # Extract title/area
        title_match = re.search(r'##\s*Title/Area\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if title_match:
            topic_data['title'] = title_match.group(1).strip()
        
        # Extract research question
        question_match = re.search(r'##\s*Research Question\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if question_match:
            topic_data['research_question'] = question_match.group(1).strip()
        
        # Extract background
        background_match = re.search(r'##\s*Background\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if background_match:
            topic_data['background'] = background_match.group(1).strip()
        
        # Extract approach
        approach_match = re.search(r'##\s*Proposed Approach\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if approach_match:
            topic_data['approach'] = approach_match.group(1).strip()
        
        # Extract keywords
        keywords_match = re.search(r'##\s*Keywords\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        if keywords_match:
            topic_data['keywords'] = keywords_match.group(1).strip()
        
        print(f"   ✅ Parsed topic: {topic_data['title'][:80]}...")
        return topic_data
    
    def generate_search_queries_with_openai(self, topic_data: Dict[str, str]) -> List[str]:
        """
        Use OpenAI to generate intelligent search queries
        
        Args:
            topic_data: Parsed research topic data
            
        Returns:
            List of search queries
        """
        if not self.openai_client:
            # Fallback: use keywords directly
            keywords = topic_data.get('keywords', '')
            return [kw.strip() for kw in keywords.split(',')][:5]
        
        print("🤖 Using OpenAI to generate search queries...")
        
        prompt = f"""Given this research topic, generate 5-7 specific search queries that would find the most relevant academic papers. 
Focus on technical terms, methods, and established research areas.

Research Title: {topic_data['title']}
Research Question: {topic_data['research_question']}
Background: {topic_data['background']}
Keywords: {topic_data['keywords']}

Generate search queries that are:
1. Specific and technical (not too broad)
2. Likely to find relevant academic papers
3. Cover different aspects of the research topic
4. Include key method names and technical terms

Return ONLY the search queries, one per line, without numbering or bullet points."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research librarian expert at finding academic papers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            queries_text = response.choices[0].message.content.strip()
            queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
            
            print(f"   ✅ Generated {len(queries)} search queries")
            for i, q in enumerate(queries, 1):
                print(f"      {i}. {q}")
            
            return queries
            
        except Exception as e:
            print(f"   ⚠️  OpenAI error: {e}")
            # Fallback to keywords
            keywords = topic_data.get('keywords', '')
            return [kw.strip() for kw in keywords.split(',')][:5]
    
    def search_semantic_scholar(self, query: str, limit: int = 5, retry_count: int = 0, max_retries: int = 3) -> List[Dict]:
        """
        Search Semantic Scholar for papers with exponential backoff
        
        Args:
            query: Search query
            limit: Maximum number of papers
            retry_count: Current retry attempt
            max_retries: Maximum retry attempts
            
        Returns:
            List of paper data
        """
        print(f"   🔍 Searching Semantic Scholar: '{query[:50]}...'")
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,authors,year,venue,abstract,citationCount,paperId,externalIds'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            # Check for rate limiting (429 status code)
            if response.status_code == 429:
                if retry_count < max_retries:
                    wait_time = (2 ** retry_count) + 1  # Exponential backoff: 2, 5, 9 seconds
                    print(f"      ⚠️  Rate limited. Waiting {wait_time} seconds before retry {retry_count + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    return self.search_semantic_scholar(query, limit, retry_count + 1, max_retries)
                else:
                    print(f"      ❌ Max retries reached. Skipping this query.")
                    return []
            
            response.raise_for_status()
            data = response.json()
            papers = data.get('data', [])
            print(f"      ✅ Found {len(papers)} papers")
            return papers
            
        except requests.exceptions.Timeout:
            print(f"      ⚠️  Request timeout")
            return []
        except Exception as e:
            print(f"      ⚠️  Error: {e}")
            return []
    
    def search_google_scholar(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search Google Scholar using scholarly
        
        Args:
            query: Search query
            limit: Maximum number of papers
            
        Returns:
            List of paper data
        """
        if not SCHOLARLY_AVAILABLE:
            return []
        
        print(f"   🔍 Searching Google Scholar: '{query[:50]}...'")
        
        papers = []
        try:
            search_query = scholarly.search_pubs(query)
            
            for i in range(limit):
                try:
                    paper = next(search_query)
                    papers.append(paper)
                    time.sleep(1)  # Be respectful to Google Scholar
                except StopIteration:
                    break
                except Exception as e:
                    print(f"      ⚠️  Error fetching paper {i+1}: {e}")
                    continue
            
            print(f"      ✅ Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            print(f"      ⚠️  Error: {e}")
            return []
    
    def convert_to_bibtex(self, papers_semantic: List[Dict], papers_scholar: List[Dict]) -> str:
        """
        Convert papers to BibTeX format
        
        Args:
            papers_semantic: Papers from Semantic Scholar
            papers_scholar: Papers from Google Scholar
            
        Returns:
            BibTeX formatted string
        """
        print("📝 Converting papers to BibTeX format...")
        
        bibtex_entries = []
        seen_titles = set()
        
        # Process Semantic Scholar papers
        for paper in papers_semantic:
            title = paper.get('title', 'N/A')
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Generate citation key
            first_author = 'unknown'
            if paper.get('authors'):
                first_author = paper['authors'][0].get('name', 'unknown').split()[-1].lower()
            year = paper.get('year', '2024')
            cite_key = f"{first_author}{year}"
            
            # Format authors
            authors = ' and '.join([a.get('name', 'Unknown') for a in paper.get('authors', [])])
            
            # Determine publication type
            venue = paper.get('venue', 'arXiv')
            arxiv_id = paper.get('externalIds', {}).get('ArXiv', '')
            
            if arxiv_id:
                bibtex = f"""@article{{{cite_key},
  title={{{title}}},
  author={{{authors}}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{{year}}}
}}"""
            else:
                bibtex = f"""@article{{{cite_key},
  title={{{title}}},
  author={{{authors}}},
  journal={{{venue}}},
  year={{{year}}}
}}"""
            
            bibtex_entries.append(bibtex)
        
        # Process Google Scholar papers
        for paper in papers_scholar:
            bib = paper.get('bib', {})
            title = bib.get('title', 'N/A')
            
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Generate citation key
            first_author = 'unknown'
            if bib.get('author'):
                authors_list = bib['author']
                if isinstance(authors_list, list) and len(authors_list) > 0:
                    first_author = authors_list[0].split()[-1].lower()
            year = bib.get('pub_year', '2024')
            cite_key = f"{first_author}{year}scholar"
            
            # Format authors
            authors = bib.get('author', ['Unknown'])
            if isinstance(authors, list):
                authors = ' and '.join(authors)
            
            bibtex = f"""@article{{{cite_key},
  title={{{title}}},
  author={{{authors}}},
  journal={{{bib.get('venue', 'N/A')}}},
  year={{{year}}}
}}"""
            
            bibtex_entries.append(bibtex)
        
        bibtex_text = '\n\n'.join(bibtex_entries)
        print(f"   ✅ Created {len(bibtex_entries)} BibTeX entries")
        
        return bibtex_text
    
    def find_references(self, topic_data: Dict[str, str], 
                       papers_per_query: int = 5,
                       use_semantic_scholar: bool = True,
                       use_google_scholar: bool = True) -> str:
        """
        Find references using multiple sources
        
        Args:
            topic_data: Research topic data
            papers_per_query: Papers to find per query
            use_semantic_scholar: Whether to use Semantic Scholar
            use_google_scholar: Whether to use Google Scholar
            
        Returns:
            BibTeX formatted references
        """
        print("\n" + "="*60)
        print("🔎 FINDING REFERENCES")
        print("="*60)
        
        # Generate search queries
        queries = self.generate_search_queries_with_openai(topic_data)
        
        all_semantic_papers = []
        all_scholar_papers = []
        
        # Search each query
        for i, query in enumerate(queries, 1):
            print(f"\n📚 Query {i}/{len(queries)}: {query}")
            
            if use_semantic_scholar:
                semantic_papers = self.search_semantic_scholar(query, papers_per_query)
                all_semantic_papers.extend(semantic_papers)
                time.sleep(1.5)  # Rate limiting: ~1 call per second
            
            if use_google_scholar and SCHOLARLY_AVAILABLE:
                scholar_papers = self.search_google_scholar(query, papers_per_query)
                all_scholar_papers.extend(scholar_papers)
                time.sleep(2)  # Rate limiting
        
        # Convert to BibTeX
        bibtex = self.convert_to_bibtex(all_semantic_papers, all_scholar_papers)
        
        print(f"\n✅ Total unique references found: {len(bibtex.split('@')) - 1}")
        
        return bibtex
    
    def generate_papers(self, topic_data: Dict[str, str], bibtex_references: str,
                       model_size: str = "12B",
                       gpu_memory_utilization: float = 0.8,
                       max_model_len: int = 40000):
        """
        Generate two papers: one without references, one with
        
        Args:
            topic_data: Research topic data
            bibtex_references: BibTeX formatted references
            model_size: Model size to use
            gpu_memory_utilization: GPU memory fraction
            max_model_len: Maximum model length
            
        Returns:
            Tuple of (paper_without_refs, paper_with_refs)
        """
        print("\n" + "="*60)
        print("🤖 INITIALIZING CYCLERESEARCHER")
        print("="*60)
        
        # Initialize CycleResearcher
        self.researcher = CycleResearcher(
            model_size=model_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        
        print("   ✅ Model loaded successfully!")
        
        # Format topic description
        topic_description = f"""
{topic_data['title']}

Research Question: {topic_data['research_question']}

Background: {topic_data['background']}

Proposed Approach: {topic_data['approach']}
"""
        
        # Generate Paper #1 (without references)
        print("\n" + "="*60)
        print("📄 GENERATING PAPER #1 (Without References)")
        print("="*60)
        print("⏱️  This may take 5-10 minutes...")
        
        papers_without_refs = self.researcher.generate_paper(
            topic=topic_description,
            references=None,
            n=1
        )
        
        print("   ✅ Paper #1 generated!")
        
        # Generate Paper #2 (with references)
        print("\n" + "="*60)
        print("📄 GENERATING PAPER #2 (With References)")
        print("="*60)
        print("⏱️  This may take 5-10 minutes...")
        
        papers_with_refs = self.researcher.generate_paper(
            topic=topic_description,
            references=bibtex_references,
            n=1
        )
        
        print("   ✅ Paper #2 generated!")
        
        return papers_without_refs[0], papers_with_refs[0]
    
    def save_results(self, topic_data: Dict, bibtex: str, 
                    paper1: Dict, paper2: Dict, output_dir: str = "output"):
        """
        Save all results to files
        
        Args:
            topic_data: Research topic data
            bibtex: BibTeX references
            paper1: Paper without references
            paper2: Paper with references
            output_dir: Output directory
        """
        print("\n" + "="*60)
        print("💾 SAVING RESULTS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save BibTeX references
        bibtex_file = output_path / "found_references.bib"
        with open(bibtex_file, 'w', encoding='utf-8') as f:
            f.write(bibtex)
        print(f"   ✅ References saved: {bibtex_file}")
        
        # Save Paper #1 (JSON)
        paper1_json = output_path / "paper_without_refs.json"
        with open(paper1_json, 'w', encoding='utf-8') as f:
            json.dump(paper1, f, indent=2)
        print(f"   ✅ Paper #1 (JSON): {paper1_json}")
        
        # Save Paper #1 (LaTeX)
        if 'latex' in paper1:
            paper1_tex = output_path / "paper_without_refs.tex"
            with open(paper1_tex, 'w', encoding='utf-8') as f:
                f.write(paper1['latex'])
            print(f"   ✅ Paper #1 (LaTeX): {paper1_tex}")
        
        # Save Paper #2 (JSON)
        paper2_json = output_path / "paper_with_refs.json"
        with open(paper2_json, 'w', encoding='utf-8') as f:
            json.dump(paper2, f, indent=2)
        print(f"   ✅ Paper #2 (JSON): {paper2_json}")
        
        # Save Paper #2 (LaTeX)
        if 'latex' in paper2:
            paper2_tex = output_path / "paper_with_refs.tex"
            with open(paper2_tex, 'w', encoding='utf-8') as f:
                f.write(paper2['latex'])
            print(f"   ✅ Paper #2 (LaTeX): {paper2_tex}")
        
        # Save comparison summary
        summary_file = output_path / "generation_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"""# Automated Research Paper Generation Results

## Research Topic
{topic_data['title']}

## Paper #1 (Without References)
**Title:** {paper1.get('title', 'N/A')}

**Abstract:**
{paper1.get('abstract', 'N/A')}

**Motivation:**
{paper1.get('motivation', 'N/A')[:500]}...

---

## Paper #2 (With References)
**Title:** {paper2.get('title', 'N/A')}

**Abstract:**
{paper2.get('abstract', 'N/A')}

**Motivation:**
{paper2.get('motivation', 'N/A')[:500]}...

---

## References Found
Total: {len(bibtex.split('@')) - 1} papers

See `found_references.bib` for full list.

---

## Files Generated
- `found_references.bib` - All found references in BibTeX format
- `paper_without_refs.json` - Paper #1 structured data
- `paper_without_refs.tex` - Paper #1 LaTeX source
- `paper_with_refs.json` - Paper #2 structured data
- `paper_with_refs.tex` - Paper #2 LaTeX source
- `generation_summary.md` - This file
""")
        print(f"   ✅ Summary: {summary_file}")
        
        print(f"\n📁 All files saved to: {output_path.absolute()}")


def main():
    """Main execution function"""
    print("="*60)
    print("🚀 AUTOMATED RESEARCH PIPELINE")
    print("="*60)
    
    # Configuration
    TOPIC_FILE = "my_research_topic.md"
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Set via export OPENAI_API_KEY=your_key
    OUTPUT_DIR = "output"
    
    # Model configuration
    MODEL_SIZE = "12B"
    GPU_MEMORY = 0.9
    MAX_MODEL_LEN = 25000
    
    # Reference search configuration
    PAPERS_PER_QUERY = 5
    USE_SEMANTIC_SCHOLAR = True
    USE_GOOGLE_SCHOLAR = True  # Set to False if scholarly not working
    
    # Initialize pipeline
    pipeline = AutomatedResearchPipeline(openai_api_key=OPENAI_API_KEY)
    
    # Step 1: Read research topic
    topic_data = pipeline.read_research_topic(TOPIC_FILE)
    
    # Step 2: Find references
    bibtex_references = pipeline.find_references(
        topic_data,
        papers_per_query=PAPERS_PER_QUERY,
        use_semantic_scholar=USE_SEMANTIC_SCHOLAR,
        use_google_scholar=USE_GOOGLE_SCHOLAR
    )
    
    # Step 3: Generate papers
    paper_without_refs, paper_with_refs = pipeline.generate_papers(
        topic_data,
        bibtex_references,
        model_size=MODEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY,
        max_model_len=MAX_MODEL_LEN
    )
    
    # Step 4: Save results
    pipeline.save_results(
        topic_data,
        bibtex_references,
        paper_without_refs,
        paper_with_refs,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETE!")
    print("="*60)
    print(f"""
Next steps:
1. Review the generated papers in: {OUTPUT_DIR}/
2. Compare paper_without_refs.tex vs paper_with_refs.tex
3. Check found_references.bib for all discovered papers
4. Compile LaTeX files to PDF if needed
5. Use CycleReviewer to evaluate the papers!

Example review command:
    from ai_researcher import CycleReviewer
    reviewer = CycleReviewer(model_size="8B")
    with open('{OUTPUT_DIR}/paper_with_refs.tex', 'r') as f:
        paper_text = f.read()
    review = reviewer.evaluate(paper_text)
    print(f"Score: {{review[0]['avg_rating']}}")
    print(f"Decision: {{review[0]['paper_decision']}}")
""")


if __name__ == "__main__":
    main() 