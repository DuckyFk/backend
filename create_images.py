from PIL import Image, ImageDraw, ImageFont

def create_sample_images():
    """Create sample images"""
    import os
    os.makedirs('../images', exist_ok=True)
    image_configs = [
        {'name': 'company_overview.png', 'text': 'Visual Alpha\nFintech Solutions', 'color': '#2563eb'},
        {'name': 'company_timeline.png', 'text': 'Founded 2019\nTokyo, Japan', 'color': '#059669'},
        {'name': 'team_structure.png', 'text': '15-20 Staff\nGlobal Team', 'color': '#dc2626'},
        {'name': 'leadership_team.png', 'text': 'Jeffrey Tsui\nCEO', 'color': '#7c3aed'},
        {'name': 'client_logos.png', 'text': 'Enterprise Clients\nInstitutional', 'color': '#ea580c'},
        {'name': 'tech_stack.png', 'text': 'NodeJS + React\nAWS + Docker', 'color': '#0891b2'},
        {'name': 'services_overview.png', 'text': 'Data Processing\nAutomation', 'color': '#be185d'}
    ]
    
    for config in image_configs:
        image_path = f"../images/{config['name']}"
        img = Image.new('RGB', (400, 200), color=config['color'])
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), config['text'], font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (400 - text_width) // 2
        y = (200 - text_height) // 2
        
        draw.text((x, y), config['text'], fill='white', font=font)
        img.save(image_path)
        print(f"✅ Created: {image_path}")

if __name__ == "__main__":
    create_sample_images()