// uploadService.js
import axios from "axios";

export const uploadImageAndGetCaption = async (imageFile) => {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await axios.post("http://backend:8000/image-captioning", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  // giả sử backend trả về { caption: "your generated caption" }
  return response.data.caption;
};
