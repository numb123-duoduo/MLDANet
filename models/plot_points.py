from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt




gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

nrows=1
figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
fig, axs = plt.subplots(nrows=1, figsize=(4.4, figh-0.3))
axs.imshow(gradient, aspect='auto', cmap='jet')
axs.set_axis_off()
plt.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.9, wspace=0,
        hspace=0)
plt.margins(0,0)

# plt.show()
fig.savefig('leg2.png')

def plot_sample_location_point(image_dir,pixel_level,pixel_index, level,sample_location_point, spatial_shape, attention_weight) :
    image = Image.open(image_dir)

    h = image.size[0]
    w = image.size[1]

    h_= spatial_shape[0]
    w_ = spatial_shape[1]

    image = image.resize(spatial_shape)
    fig=plt.figure(figsize=(5,5))
    
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_ticks_position('top')
    plt.imshow(image)
    plt.axis('off')

    color = np.array([1, 0.0, 0.0])
    b = torch.FloatTensor(attention_weight)

    
    
 

    sorted, indices = torch.sort(b)#由小到大

    x_list=[]
    y_list=[]
    for i in range(len(sample_location_point)):
        # x = sample_location_point[indices[i]][0]*h_*(h/h_)
        # y = sample_location_point[indices[i]][1]*w_*(w/w_)
        x_list.append((sample_location_point[indices[i]][0]*(h_)-0.5).detach().cpu().numpy().item())
        y_list.append((sample_location_point[indices[i]][1]*(w_)-0.5).detach().cpu().numpy().item())
        #print ('attention_weight[i]',attention_weight[i])
        # plt.plot(x.item(), y.item(), 'o', color='red')
        # plt.plot(x.item(), y.item(), 'o', color=(i+1)/len(sample_location_point)* color,clip_on=False)
    plt.scatter(x_list,y_list,s=80,c=list(range(len(sample_location_point))),cmap='Reds')
    # plt.scatter(x_list,y_list,c=list(range(len(sample_location_point))),cmap='gist_heat')
    # plt.colorbar(extend="max")
    # for i in range(len(sample_location_point)):
    #     # x = sample_location_point[indices[i]][0]*h_*(h/h_)
    #     # y = sample_location_point[indices[i]][1]*w_*(w/w_)
    #     x = sample_location_point[indices[i]][0]*(h_)-0.5
    #     y = sample_location_point[indices[i]][1]*(w_)-0.5
    #     #print ('attention_weight[i]',attention_weight[i])
    #     # plt.plot(x.item(), y.item(), 'o', color='red')
    #     plt.plot(x.item(), y.item(), 'o', color=(i+1)/len(sample_location_point)* color,clip_on=False)
    
    

    if (pixel_level==0) & (level==0):
        py= (pixel_index//128)-1
        px=(pixel_index%128)-1
        plt.plot(px,py, 'b^',markersize=13)
    if (pixel_level==3) & (level==3):
        py= (pixel_index-21504)//16-1
        px=(pixel_index-21504)%16-1
        plt.plot(px,py, 'b^',markersize=13)
    if (pixel_level==2) & (level==2):
        py= (pixel_index-20480)//32-1
        px=(pixel_index-20480)%32-1
        plt.plot(px,py, 'b^',markersize=13)
    if (pixel_level==1) & (level==1):
        py= (pixel_index-16384)//64-1
        px=(pixel_index-16384)%64-1
        plt.plot(px,py, 'b^',markersize=13)
        
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,
        hspace=0)
    plt.margins(0,0)
    plt.savefig('level'+str(pixel_level)+'point'+str(level)+'.png')

