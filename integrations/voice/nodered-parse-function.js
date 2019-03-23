let obj = msg.payload;
if(obj.detections === 0) {
    msg.payload = "I can't find anything";
} else {
    let objectType = 'gummies';
    let threshold = .80
    let types =  {}
    obj.tags.forEach(tag =>{
        nm = tag['tag'];
        if(tag['score'] > threshold) { 
            if(types.hasOwnProperty(nm)) {
                types[nm] +=1
            } else {
                types[nm] = 1
            }
        }
    })
    console.log(types);
    var keys = Object.keys(types);
    console.log(keys);
    let rslt = 'I found';
    for(i = 0; i < keys.length; i++) {
        console.log(keys[i]);
        let ct = types[keys[i]]
        
        let vr = ' ' + ct + ' ' + keys[i] + ' ' + objectType
        if(i === 0 && i+1 === keys.length) {
            rslt += vr 
        }else if(i + 1 == keys.length) {
            rslt += ' and ' +vr
        }else {
            rslt += ', ' +vr
        }
    }
    msg.payload = rslt
}
return msg;