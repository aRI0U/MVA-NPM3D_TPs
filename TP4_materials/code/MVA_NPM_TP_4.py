import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
import time

METHODS = ['bp', 'mf', 'diffuse']
METHOD = sys.argv[1].lower() if len(sys.argv) > 1 else 'bp'
METHOD = METHOD if METHOD in METHODS else 'bp'
FNAME = 'normal.png'

class Material:
    def __init__(
        self, albedo=None,
        diffuse_coeff=None,
        specular_coeff=None,
        shininess_coeff=None,
        alpha=None,
        F0=None
    ):
        self.albedo = np.array(albedo)
        self.diffuse_coeff = diffuse_coeff
        self.specular_coeff = specular_coeff
        self.shininess_coeff = shininess_coeff
        self.alpha = alpha
        self.F0 = F0

class LightSource:
    def __init__(self, pos, color, intensity):
        self.pos = np.array(pos)
        self.color = np.array(color)
        self.intensity = intensity


def shade(normalImage, mat, sources):
    normals = np.array(normalImage)
    H, W, C = normals.shape

    reflectance = mat.diffuse_coeff * mat.albedo / np.pi

    # compute positions of all pixels
    pixelPos = np.zeros(normals.shape) # pixel_pos[i,j] = [i,j,0]
    pixelPos[...,0], pixelPos[...,1] = np.meshgrid(np.linspace(0, H/W, H), np.linspace(0, 1, W), indexing='ij')

    incoming = np.zeros((len(sources),) + normals.shape)
    for i, s in enumerate(sources):
        incoming[i] = s.pos - pixelPos

    projection = np.sum(incoming * normals, axis=-1).transpose((1,2,0)) # shape: (H, W, S)

    lights = np.array([
        s.intensity * s.color for s in sources
    ]) # shape: (S, C)

    rgbImage = reflectance * (projection @ lights)

    rgbMin, rgbMax = rgbImage.min(), rgbImage.max()

    rgbImage = 255*(rgbImage - rgbMin)/(rgbMax - rgbMin)

    return Image.fromarray(rgbImage.astype(np.uint8))




def shadeBP(normalImage, mat, sources, sensor):
    normals = np.array(normalImage)
    H, W, C = normals.shape

    reflectance_d = mat.diffuse_coeff * mat.albedo / np.pi # shape: (C,)

    # compute positions of all pixels
    pixelPos = np.zeros(normals.shape) # pixel_pos[i,j] = [i,j,0]
    pixelPos[...,0], pixelPos[...,1] = np.meshgrid(np.linspace(0, H/W, H), np.linspace(0, 1, W), indexing='ij')

    incoming = np.zeros((len(sources),) + normals.shape) # omega_i
    for i, s in enumerate(sources):
        incoming[i] = s.pos - pixelPos

    outgoing = pixelPos - sensor # omega_o

    halfvector = incoming + outgoing # shape: (S, H, W, C)
    halfvector /= np.linalg.norm(halfvector, axis=-1)[...,np.newaxis]

    proj_i = np.sum(incoming * normals, axis=-1).transpose(1,2,0) # shape: (H, W, S)
    proj_h = np.sum(halfvector * normals, axis=-1).transpose(1,2,0) # shape: (H, W, S)

    reflectance_s = mat.specular_coeff * proj_h**mat.shininess_coeff

    lights = np.array([
        s.intensity * s.color for s in sources
    ]) # shape: (S, C)

    rgbImage = reflectance_d * (proj_i @ lights) + (reflectance_s * proj_h) @ lights

    rgbMin, rgbMax = rgbImage.min(), rgbImage.max()

    rgbImage = 255*(rgbImage - rgbMin)/(rgbMax - rgbMin)

    return Image.fromarray(rgbImage.astype(np.uint8))




def shadeMF(normalImage, mat, sources, sensor):
    normals = np.array(normalImage)
    H, W, C = normals.shape

    reflectance_d = mat.diffuse_coeff * mat.albedo / np.pi # shape: (C,)

    # compute positions of all pixels
    pixelPos = np.zeros(normals.shape) # pixel_pos[i,j] = [i,j,0]
    pixelPos[...,0], pixelPos[...,1] = np.meshgrid(np.linspace(0, H/W, H), np.linspace(0, 1, W), indexing='ij')

    incoming = np.zeros((len(sources),) + normals.shape) # omega_i
    for i, s in enumerate(sources):
        incoming[i] = s.pos - pixelPos

    outgoing = pixelPos - sensor # omega_o, shape: (H, W, C)

    halfvector = incoming + outgoing # shape: (S, H, W, C)
    halfvector /= np.linalg.norm(halfvector, axis=-1)[...,np.newaxis]

    # scalar products n.omega_i and n.omega_h
    proj_i = np.sum(incoming * normals, axis=-1).transpose(1,2,0) # shape: (H, W, S)
    proj_h = np.sum(halfvector * normals, axis=-1).transpose(1,2,0) # shape: (H, W, S)
    proj_o = np.sum(outgoing * normals, axis=-1)[...,np.newaxis] # shape: (H, W, 1)

    # reflectance
    microfacetDistribution = (mat.alpha / (1 + (mat.alpha**2 - 1)*(proj_h)**2))**2/np.pi # shape: (H,W,S)

    scal_ih = np.sum(incoming * halfvector, axis=-1).transpose(1,2,0)
    fresnelTerm = mat.F0 + (1-mat.F0) * 2**((-5.55473*proj_h-6.98613) * proj_h) # shape: (H, W, S)

    k = (np.sqrt(mat.alpha)+1)**2/8
    geometric_i = proj_i / (proj_i*(1-k) + k)
    geometric_o = proj_o / (proj_o*(1-k) + k)
    geometricTerm = geometric_i * geometric_o

    reflectance_s = microfacetDistribution * fresnelTerm * geometricTerm \
                    / (4 * proj_i * proj_o)
    reflectance_s[np.isnan(reflectance_s)] = 0

    lights = np.array([
        s.intensity * s.color for s in sources
    ]) # shape: (S, C)

    rgbImage = reflectance_d * (proj_i @ lights) + (reflectance_s * proj_h) @ lights
    rgbMin, rgbMax = rgbImage.min(), rgbImage.max()
    rgbImage[rgbImage>0.0005*rgbMax] = (rgbImage * (rgbImage<0.0005*rgbMax)).max()
    rgbImage[rgbImage<0.001*rgbMin] = (rgbImage * (rgbImage>0.001*rgbMin)).min()
    rgbMin, rgbMax = rgbImage.min(), rgbImage.max()
    # rgbImage[rgbImage]
    # print(rgbMax, rgbMin)
    # plt.hist(rgbImage.reshape(-1), bins=50)
    # plt.show()
    rgbImage = 255*(rgbImage - rgbMin)/(rgbMax - rgbMin)

    return Image.fromarray(rgbImage.astype(np.uint8))

if __name__ == '__main__':
    normalImage = Image.open(FNAME).convert('RGB')

    mat = Material(
        albedo=[0.3, 0.8, 0.5],
        diffuse_coeff=0.1,
        specular_coeff=2,
        shininess_coeff=1,
        alpha=0.5,
        F0=0.2
    )
    sources = [
        LightSource([0, 1, 1], [1, 1, 0.3], 1),
        LightSource([1, 1, 2], [1, 1, 1], 0.2)
    ]
    sensor = [1, 1, 1]

    start = time.time()

    if METHOD == 'mf':
        renderImage = shadeMF(normalImage, mat, sources, sensor)
    elif METHOD == 'bp':
        renderImage = shadeBP(normalImage, mat, sources, sensor)
    else:
        renderImage = shade(normalImage, mat, sources)

    end = time.time()

    plt.imshow(renderImage)
    plt.show()

    renderImage.save('render.png')
    print('Done. Time elapsed: {:.3f}s'.format(end-start))
