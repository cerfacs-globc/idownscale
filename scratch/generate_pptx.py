import os
import re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE

def create_pptx(md_file, output_file):
    prs = Presentation()
    
    # 16:9 Aspect Ratio (Widescreen)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    with open(md_file, 'r') as f:
        content = f.read()

    # Split into slides
    slides_raw = re.split(r'\n---', content)
    
    slide_count = 0
    for slide_text in slides_raw:
        slide_text = slide_text.strip()
        if not slide_text or slide_text.startswith('# EGU'):
            continue
            
        # Extract metadata
        lines = slide_text.split('\n')
        title = "Untitled Slide"
        bullets = []
        speaker_notes = ""
        images = []
        
        in_notes = False
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Slide Title (Supports ## and ###)
            if re.match(r'^###?\s*Slide', line):
                title = line.split(':', 1)[1].strip() if ':' in line else line
                # Remove common slide-duration indicators like (2 mins)
                title = re.sub(r'\(\d+\s*min(s)?\)', '', title).strip()
                continue
                
            # Speaker Notes
            if line.startswith('**Speaker Notes:**'):
                in_notes = True
                speaker_notes += line.replace('**Speaker Notes:**', '').strip() + ' '
                continue
            if in_notes:
                if line.endswith('---') or line.startswith('**Keywords:**'): 
                    in_notes = False
                    continue
                speaker_notes += line + ' '
                continue
                
            # Images
            img_match = re.search(r'!\[.*?\]\((.*?)\)', line)
            if img_match:
                images.append(img_match.group(1))
                continue
                
            # Bullets (Supports * and -)
            if line.startswith('*') or line.startswith('-'):
                bullets.append(line[1:].strip())
        
        # Create Slide
        slide_layout = prs.slide_layouts[1] # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        
        # Set Title
        slide.shapes.title.text = title
        
        # Set Bullets
        body_shape = slide.placeholders[1]
        tf = body_shape.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        tf.text = "" # Clear placeholder
        
        # Font scaling logic
        font_size = Pt(24)
        if len(bullets) > 5:
            font_size = Pt(20)
        if len(bullets) > 8:
            font_size = Pt(16)

        for bullet in bullets:
            p = tf.add_paragraph()
            p.text = bullet
            p.level = 0
            p.font.size = font_size
            
        # Add Speaker Notes
        notes_slide = slide.notes_slide
        notes_slide.notes_text_frame.text = speaker_notes.strip()
        
        # Add Images (Positioned at bottom right area)
        if images:
            # Shift body shape to the left if there are images
            body_shape.width = Inches(8)
            
            for i, img_path in enumerate(images):
                if os.path.exists(img_path):
                    # Vertical stacking helper
                    top = Inches(2.0 + (i * 2.5))
                    left = Inches(8.5)
                    try:
                        slide.shapes.add_picture(img_path, left, top, height=Inches(2.5))
                    except Exception as e:
                        print(f"Error adding image {img_path}: {e}")
                else:
                    print(f"Warning: Image not found at {img_path}")

        slide_count += 1
        print(f"Processed Slide {slide_count}: {title}")

    prs.save(output_file)
    print(f"\nSUCCESS: Presentation saved to {output_file}")

if __name__ == "__main__":
    md_path = "/scratch/globc/page/idownscale_garcia_clean/EGU_Presentation_Slides.md"
    out_path = "/scratch/globc/page/idownscale_garcia_clean/EGU26_Downscaling_ML.pptx"
    create_pptx(md_path, out_path)
