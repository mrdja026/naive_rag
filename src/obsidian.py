import re
from typing import Callable, Optional
from pathlib import Path


def chunk_by_headers_obsidian(markdown_text: str, metadata_fn: Optional[Callable[[str], dict]] = None):
    """Split Obsidian markdown by headers, keeping each section intact.

    metadata_fn: optional callable that accepts a chunk string and returns a dict of metadata.
                 If None, `extract_metadata` will be used at runtime.
    By default this splits on H1, H2 and H3. Lower-level headers update the header
    hierarchy but remain inside the same chunk.
    """
    # avoid using extract_metadata as a default in the signature (not defined yet)
    if metadata_fn is None:
        metadata_fn = extract_metadata

    # by default chunk on H1, H2 and H3
    chunk_levels = [1, 2, 3]

    lines = markdown_text.split('\n')
    chunks = []
    current_chunk = []
    current_headers = {}

    for line in lines:
        # Check if line is a header
        if line.strip().startswith('#'):
            header_level = len(line) - len(line.lstrip('#'))
            header_text = line.strip('#').strip()

            # If this header level should start a new chunk, emit the previous chunk (if any)
            if header_level in chunk_levels:
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if current_headers:
                        # previous chunk belongs to the previously-seen header -> save it
                        if chunk_content:
                            chunks.append({
                                'content': chunk_content,
                                'headers': current_headers.copy(),
                                'metadata': metadata_fn(chunk_content)
                            })
                        # start a fresh chunk for the new header
                        current_chunk = [line]
                    else:
                        # no header seen yet: keep preface lines and include this header line so
                        # the preface attaches to the first header
                        current_chunk = current_chunk + [line]
                else:
                    # no pending lines, start new chunk with the header
                    current_chunk = [line]

                # Update header hierarchy for this header
                current_headers[f'h{header_level}'] = header_text
                for i in range(header_level + 1, 7):
                    current_headers.pop(f'h{i}', None)
            else:
                # Header level is not a chunk boundary: update hierarchy but keep the line
                # inside the current chunk.
                current_headers[f'h{header_level}'] = header_text
                for i in range(header_level + 1, 7):
                    current_headers.pop(f'h{i}', None)

                if current_chunk:
                    current_chunk.append(line)
                else:
                    # no pending lines, start a chunk that will include this header
                    current_chunk = [line]
        else:
            current_chunk.append(line)

    # Don't forget the last chunk
    if current_chunk:
        chunk_content = '\n'.join(current_chunk).strip()
        # include any non-empty last chunk (removed 50-character threshold)
        if chunk_content:
            chunks.append({
                'content': chunk_content,
                'headers': current_headers.copy(),
                'metadata': metadata_fn(chunk_content)
            })

    return chunks


def extract_metadata(chunk_text: str) -> dict:
    # Find Obsidian tags like #tag
    tags = re.findall(r'#(\w+)', chunk_text)
    
    # Find external URLs
    urls = re.findall(r'https?://[^\s)]+', chunk_text)
    
    # Find Obsidian internal links [[Note]]
    internal_links = re.findall(r'\[\[([^\]]+)\]\]', chunk_text)
    
    return {
        'tags': list(set(tags)),
        'urls': list(set(urls)),
        'internal_links': list(set(internal_links))
    }
if __name__ == '__main__':
    # load test note from obsidian_assets/main_discovery.txt next to the repo
    assets_path = Path(__file__).resolve().parent / 'obsidian_assets' / 'main_discovery.txt'
    if assets_path.exists():
        obsidian_note = assets_path.read_text(encoding='utf-8')
    else:
        obsidian_note = ''

    # default behavior using extract_metadata
    print('Default metadata_fn:')
    print(chunk_by_headers_obsidian(obsidian_note))

    # custom metadata function
    def my_meta(text: str) -> dict:
        return {'length': len(text), 'example': True}

    print('\nCustom metadata_fn:')
    print(chunk_by_headers_obsidian(obsidian_note, metadata_fn=my_meta))