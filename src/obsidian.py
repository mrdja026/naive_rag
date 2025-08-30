import re
from typing import Callable, Optional
from pathlib import Path


def chunk_by_headers_obsidian(markdown_text: str, metadata_fn: Optional[Callable[[str], dict]] = None):
    """Split Obsidian markdown by headers, keeping each section intact.

    metadata_fn: optional callable that accepts a chunk string and returns a dict of metadata.
                 If None, `extract_metadata` will be used at runtime.
    """
    # avoid using extract_metadata as a default in the signature (not defined yet)
    if metadata_fn is None:
        metadata_fn = extract_metadata

    lines = markdown_text.split('\n')
    chunks = []
    current_chunk = []
    current_headers = {}

    for line in lines:
        # Check if line is a header
        if line.strip().startswith('#'):
            # If we already have a header context, the current_chunk belongs to that header and
            # should be emitted. If we have lines before the first header (no current_headers),
            # attach them to the upcoming header instead of emitting them as a separate chunk.
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
                    # no header seen yet: keep preface lines and include this header line as part
                    # of the same chunk so the preface attaches to the first header
                    current_chunk = current_chunk + [line]
            else:
                # no pending lines, start new chunk with the header
                current_chunk = [line]

            # Start new header context
            header_level = len(line) - len(line.lstrip('#'))
            header_text = line.strip('#').strip()

            # Update header hierarchy
            current_headers[f'h{header_level}'] = header_text
            # Clear lower-level headers
            for i in range(header_level + 1, 7):
                current_headers.pop(f'h{i}', None)
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