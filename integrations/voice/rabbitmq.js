const amq = require('amqplib')

async function runme() {
    let msg = { 'type': 'count', 'request': {} }
    amq.connect('amqp://user:Z7xBMdauGT@35.222.194.94').then(conn => {
        var ok = conn.createChannel();
        ch = ok.then(function(ch) {
            ch.purgeQueue("dialogflow_response")
            console.log('purged');
            return ch
        });
        return ch;
    }).then(ch =>{
        
        ch.assertQueue('dialogflow_request')
        ch.sendToQueue('dialogflow_request', Buffer.from(JSON.stringify(msg)))
        ch.assertQueue('dialogflow_response')
        ch.prefetch(1)
        return ch;
    }).then(ch =>{
        msg =ch.consume('dialogflow_response',msg =>{
            console.log(msg.content.toString())
            ch.ack(msg)
            ch.close();
        })
        return ch;
    }).catch(err => {
        console.log(err);
    })
        
    
    
}

runme();