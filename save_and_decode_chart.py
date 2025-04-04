import base64
import os
from typing import Optional
from pydantic import BaseModel, Field

class ImageProcessor(BaseModel):
    """
    A class to handle encoding and decoding of base64 images
    """
    base64_string: str = Field(..., description="Base64 encoded image string")
    output_path: str = Field("etc_chart.png", description="Path to save the decoded image")
    
    def save_base64_to_file(self, filename: str = "etc_chart_base64.txt") -> str:
        """
        Save the base64 string to a file
        
        Args:
            filename: Name of the file to save the base64 string to
            
        Returns:
            Path to the saved file
        """
        with open(filename, "w") as f:
            f.write(self.base64_string)
        return filename
    
    def decode_and_save_image(self) -> str:
        """
        Decode the base64 string and save as an image
        
        Returns:
            Path to the saved image
        """
        # Remove header if present (e.g., "data:image/png;base64,")
        if "," in self.base64_string:
            self.base64_string = self.base64_string.split(",", 1)[1]
            
        # Decode the base64 string
        image_data = base64.b64decode(self.base64_string)
        
        # Save the decoded data to a file
        with open(self.output_path, "wb") as f:
            f.write(image_data)
            
        return self.output_path

def process_from_file(base64_file: str, output_image: str = "etc_chart.png") -> Optional[str]:
    """
    Process a base64 file and convert it to an image
    
    Args:
        base64_file: Path to the file containing base64 encoded image
        output_image: Path to save the decoded image
        
    Returns:
        Path to the saved image or None if error occurs
    """
    try:
        with open(base64_file, "r") as f:
            base64_string = f.read().strip()
        
        processor = ImageProcessor(base64_string=base64_string, output_path=output_image)
        return processor.decode_and_save_image()
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Paste your base64 string here
    base64_string = """
    PUT_YOUR_BASE64_STRING_HERE
    """
    
    # Strip whitespace
    base64_string = base64_string.strip()
    
    # Create processor
    processor = ImageProcessor(base64_string=base64_string)
    
    # Save base64 to file
    base64_file = processor.save_base64_to_file()
    print(f"Base64 saved to {base64_file}")
    
    # Decode and save image
    image_path = processor.decode_and_save_image()
    print(f"Image saved to {image_path}")
    
    print("\nAlternatively, if you've already saved the base64 to a file:")
    print("python save_and_decode_chart.py")
    
    # Check if image exists
    if os.path.exists(image_path):
        print(f"\nSuccess! Image saved to {image_path}")
    else:
        print("\nError: Image not saved correctly") 