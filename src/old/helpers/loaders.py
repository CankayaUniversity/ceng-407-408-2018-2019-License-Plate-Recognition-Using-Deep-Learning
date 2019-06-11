import json


class Parser(object):
    
    def parse(self, path):
        raw = {}
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            parsedData = self.process(raw)
            return parsedData
        
        except (IOError, AttributeError) as e:
            print (str(e))
            return None
        
    def process(self, raw):
        pass
    
class VggObjExistParser(Parser):
    def process(self, raw):
        data = raw['_via_img_metadata']
        tablu = {}
        for key, value in data.items():
            tablu[value["filename"]] = value['regions']
        data = tablu
        cleaned_data = {}
        for key, value in data.items():
             if len(value) > 0:
                 cleaned_data[key] = value
            

        polished = {}
        for key, value in cleaned_data.items():
            arr = []
            for el in value:
                el = el['shape_attributes']
                arr.append({'x':el['x'], 'y':el['y'], 'w':el['width'],'h':el['height']})
            polished[key] = arr

        return polished
        