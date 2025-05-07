// ImageCaptioning.jsx
import React, { useState } from "react";
import { uploadImageAndGetCaption } from "./uploadService";


const ImageCaptioning = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) {
      alert("Please select an image first.");
      return;
    }
    setLoading(true);
    try {
      const captionResult = await uploadImageAndGetCaption(selectedImage);
      setCaption(captionResult);
    } catch (error) {
      console.error(error);
      alert("Failed to get caption.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Image Captioning</h1>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      {selectedImage && (
        <img
          src={URL.createObjectURL(selectedImage)}
          alt="Selected"
          className="image"
        />
      )}
      <button
        onClick={handleSubmit}
      >
        {loading ? "Generating Caption..." : "Generate Caption"}
      </button>
      {caption && (
        <div>
          <h2>Caption:</h2>
          <p>{caption}</p>
        </div>
      )}
    </div>
  );
};

export default ImageCaptioning;
