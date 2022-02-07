import pickle
import os
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    # Path = 'Final/Data/'
    # filelist = os.listdir(Path)
    # PubmedDoc_All = {}
    # for filename in filelist:
    #     PubmedDoc = {}
    #     if filename.endswith('.xml'):
    #         tree = ET.parse(f'{Path}/{filename}')
    #         for art in tree.iter(tag = 'Article'):
    #             Article = {}
    #             Title = art.find("ArticleTitle").text
    #             for abs in art.iter(tag = 'Abstract'):    
    #                 for content in abs.iter(tag= 'AbstractText'):
    #                     if 'Label' in content.attrib and content.text is not None:
    #                         Article[content.attrib['Label']] = content.text
    #                     elif content.text is not None:
    #                         Article['Abstract'] = content.text
    #             if len(Article)!=0:
    #                 PubmedDoc[Title] = Article
    #                 PubmedDoc_All[Title] = Article
    #             if len(PubmedDoc)==1000:
    #                 break

    #         print(len(PubmedDoc))

    #         with open(f'{Path}/{filename[:-4]}.pkl', 'wb')as fpick:
    #             pickle.dump(PubmedDoc, fpick)

    # print(len(PubmedDoc_All))
    # with open(f'{Path}/Full.pkl', 'wb')as fpick:
    #     pickle.dump(PubmedDoc_All, fpick)
    Path = 'Final/Data/'
    filelist = os.listdir(Path)
    PubmedDoc_All = {}
    Category = {}
    for filename in filelist:
        PubmedDoc = {}
        cate = filename[:-9]
        if filename.endswith('.xml'):
            tree = ET.parse(f'{Path}/{filename}')
            for art in tree.iter(tag = 'Article'):
                Article = {}
                Title = art.find("ArticleTitle").text
                for abs in art.iter(tag = 'Abstract'):    
                    for content in abs.iter(tag= 'AbstractText'):
                        if 'Label' in content.attrib and content.text is not None:
                            Article[content.attrib['Label']] = content.text
                        elif content.text is not None:
                            Article['Abstract'] = content.text
                if len(Article)!=0:
                    PubmedDoc[Title] = Article
                    PubmedDoc_All[Title] = Article
                    Category[Title] = cate
                if len(PubmedDoc)==1000:
                    break

            print(len(PubmedDoc))
    

            with open(f'{Path}/{filename[:-4]}.pkl', 'wb')as fpick:
                pickle.dump(PubmedDoc, fpick)

    print(len(PubmedDoc_All))
    with open(f'{Path}/Full.pkl', 'wb')as fpick:
        pickle.dump(PubmedDoc_All, fpick)

    print(len(Category))
    with open(f'{Path}/Category.pkl', 'wb')as fpick:
        pickle.dump(Category, fpick)