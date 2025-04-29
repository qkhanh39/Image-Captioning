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
    <div className="flex flex-col items-center gap-4 p-6">
      <h1 className="text-2xl font-bold">Image Captioning</h1>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      {selectedImage && (
        <img
          src={URL.createObjectURL(selectedImage)}
          alt="Selected"
          className="w-64 h-64 object-cover rounded"
        />
      )}
      <button
        onClick={handleSubmit}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        {loading ? "Generating Caption..." : "Generate Caption"}
      </button>
      {caption && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <h2 className="text-lg font-semibold">Caption:</h2>
          <p>{caption}</p>
        </div>
      )}
    </div>
  );
};

export default ImageCaptioning;
