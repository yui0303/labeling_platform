from xml.etree import ElementTree as ET
from xml.dom import minidom
import os
import shutil

from requests import patch

def create_xml(img_folder, img_name, h, w, x1, y1, x2, y2, obj_name):
        annotation = ET.Element('annotation')

        folder = ET.SubElement(annotation,'folder')
        folder.text=(img_folder)

        filename = ET.SubElement(annotation,'filename')
        filename.text = img_name

        path = ET.SubElement(annotation,'path')
        path.text = os.getcwd()+'\\'+img_folder+'\\'+img_name

        source = ET.SubElement(annotation,'source')
        database = ET.SubElement(source,'database')
        database.text = "Unknown"

        size = ET.SubElement(annotation,'size')

        width = ET.SubElement(size,'width')
        width.text = str(w)
        height = ET.SubElement(size,'height')
        height.text = str(h)
        depth = ET.SubElement(size,'depth')
        depth.text = '3'
        
        segmented = ET.SubElement(annotation,'segmented')
        segmented.text = '0'

        for i in range(len(x1)):
            _object = ET.SubElement(annotation,'object')
            
            name = ET.SubElement(_object,'name')
            name.text = str(obj_name[i])
            pose = ET.SubElement(_object,'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(_object,'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(_object,'difficult')
            difficult.text = '0'
            
            bndbox = ET.SubElement(_object,'bndbox')
            xmin = ET.SubElement(bndbox,'xmin')
            xmin.text = '%d'%x1[i]
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = '%d'%y1[i]
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = '%d'%x2[i]
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = '%d'%y2[i]
        
        return annotation

def toxml():
    img_name_list=[]
    path = os.listdir('txtArea')
    for n in path:
        if n.endswith(').txt'):
            os.remove('txtArea/'+n.split('.')[0]+'.jpg.txt')
            shutil.move('txtArea/'+n,'txtArea/'+n.split('.')[0]+'.jpg.txt')
    path = os.listdir('txtArea')
    for n in path:
        if not(n.endswith('txt')):
            continue
        f = open('txtArea/'+n)
        text = []
        for line in f:
            
            text.append(line)


        name=text[0]
        name=name[name.rfind("= ")+2:-2]
        #append img name to a list
        img_name_list.append(name)
        width=text[1]
        width=width[width.rfind("= ")+2:-2]
        height=text[2]
        height=height[height.rfind("= ")+2:-2]
        R=text[3]
        R=float(R[R.rfind("= ")+2:-2])
        f.close()


        f = open('txtArea/'+n)
        lines = f.readlines()[5:-2]
        label_list=[]
        numofobj=len(lines)//7

        f.close()

        for i in range(0,numofobj):
            label_element=[]
            for j in range(i*7,i*7+7):
                if ( j%7==1  ):
                    label_element.append(round(int(lines[j][11:-2])*R))
                elif(j%7==2):
                    label_element.append(round(int(lines[j][11:-2])*R))
                elif ( j%7==3 ):
                    label_element.append(round(int(lines[j][13:-2])*R))
                elif(j%7==4):
                    label_element.append(round(int(lines[j][14:-2])*R))
                elif ( j%7==5  ):
                    label_element.append(lines[j][14:-2])
            print(label_element)
            #label_element.pop()
            label_list.append(label_element)
        #xmin,ymin,xmax,ymax

        for i in label_list:
            i[2]+=i[0]
            i[3]+=i[1]
            # tem=i[0]
            # i[0]=i[1]
            # i[1]=tem

            # tem=i[2]
            # i[2]=i[3]
            # i[3]=tem
        

        label_list=[list(i) for i in zip(*label_list)]
        print(label_list)
        print(n)
        annotation = create_xml(name[:name.rfind('/')], name[name.rfind('/')+1:], height, width, label_list[0], label_list[1], label_list[2], label_list[3], label_list[4])
        tree = ET.ElementTree(annotation) 
        xmlstr = minidom.parseString(ET.tostring(tree._root)).toprettyxml(indent="   ")
        with open('prepare_data/with_annotation/train/'+n.split('_')[-1].split('.')[0]+'.xml', "w") as f:
            f.write(xmlstr)
        shutil.move('txtArea/'+n,'txtTemp/'+n)
        #os.remove('txtArea/'+n)
    #print(img_name_list)

    for i in img_name_list:
        x=i[:-3]+'xml'
        if os.path.isfile(x):
            os.remove(x)
        dest='prepare_data/with_annotation/train/'+i.split('/')[-1]
        print(dest)
        shutil.move(i,dest)
        

#toxml()