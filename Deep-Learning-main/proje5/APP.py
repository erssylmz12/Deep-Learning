import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2

#Cihaz ayarı
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Eğitilmiş modeli aktar
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # 10 yara sınıfı
model.load_state_dict(torch.load('best_wound_resnet18.pt', map_location=device))
model.to(device)
model.eval()

#Sınıf isimleri
class_names = [
    "Abrasions", "Bruises", "Burns", "Cut", "Diabetic Wounds",
    "Laceration", "Normal", "Pressure Wounds", "Surgical Wounds", "Venous Wounds"
]

#Önişleme dönüşümü
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#Grad-CAM implementasyonu
def generate_gradcam(model, image_tensor, target_class=None):
    features, gradients = [], []
    
    def forward_hook(module, inp, out):
        features.append(out)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    final_conv = model.layer4[1].conv2
    fwd = final_conv.register_forward_hook(forward_hook)
    bwd = final_conv.register_backward_hook(backward_hook)
    
    #Forward pass
    tensor = image_tensor.unsqueeze(0).to(device)
    output = model(tensor)
    
    #Hedef sınıfı belirle
    if target_class is None:
        target_class = output.argmax().item()
    
    #Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    #CAMı hesapla
    grad = gradients[0].mean(dim=[2, 3], keepdim=True)
    fmap = features[0]
    cam = (fmap * grad).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    
    fwd.remove()
    bwd.remove()
    return cam, target_class

#Streamlit UI
st.title("Wound Classification & Localization")
st.write("Upload an image of a wound; the model will predict its type and highlight the region.\nThe model also handles 'Normal' cases where no wound is detected.")

uploaded_file = st.file_uploader("Choose a wound image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Analyze"):
        input_tensor = transform(image)
        cam, pred = generate_gradcam(model, input_tensor)
        label = class_names[pred]
        
        if label == "Normal":
            st.success("No wound detected (Normal).")
        else:
            #Isı haritası overlayini oluştur
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            img_np = np.array(image.resize((224, 224)))
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
            
            st.image(overlay, caption=f'Predicted: {label}', use_column_width=True)
