import re

def split_by_triple_newline(text):
    chunks = re.split(r'\n{3,}', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def split_into_chunks(text, min_chunk_size=1, max_chunk_size=1000):
    text = text.strip()
    if not text:
        return []
        
    pre_chunks = split_by_triple_newline(text)
    final_chunks = []
    
    for pre_chunk in pre_chunks:
        if min_chunk_size <= len(pre_chunk) <= max_chunk_size:
            final_chunks.append(pre_chunk)
            continue
            
        if len(pre_chunk) < min_chunk_size:
            if len(pre_chunks) == 1:
                final_chunks.append(pre_chunk)
            continue
            
        if len(pre_chunk) > max_chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', pre_chunk)
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_size = len(sentence)
                
                if current_size + sentence_size <= max_chunk_size:
                    current_chunk.append(sentence)
                    current_size += sentence_size
                else:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text) >= min_chunk_size:
                            final_chunks.append(chunk_text)
                    current_chunk = [sentence]
                    current_size = sentence_size
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    final_chunks.append(chunk_text)
    
    return final_chunks

def chunk_markdown(content):
    sections = re.split(r'(^#{1,6}\s.*$)', content, flags=re.MULTILINE)
    chunks = []
    current_header = ""
    current_text = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if re.match(r'^#{1,6}\s', section):
            if current_text:
                section_chunks = split_into_chunks(current_text)
                for chunk in section_chunks:
                    full_chunk = f"{current_header}\n{chunk}" if current_header else chunk
                    chunks.append(full_chunk.strip())
            current_header = section
            current_text = ""
        else:
            current_text += "\n" + section
    
    if current_text:
        section_chunks = split_into_chunks(current_text)
        for chunk in section_chunks:
            full_chunk = f"{current_header}\n{chunk}" if current_header else chunk
            chunks.append(full_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]
