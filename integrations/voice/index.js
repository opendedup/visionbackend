// See https://github.com/dialogflow/dialogflow-fulfillment-nodejs
// for Dialogflow fulfillment library docs, samples, and to report issues
'use strict';

const functions = require('firebase-functions');
const { WebhookClient } = require('dialogflow-fulfillment');
const { Connection } = require('amqplib-as-promised')


process.env.DEBUG = 'dialogflow:debug'; // enables lib debugging statements

exports.dialogflowRequest = functions.https.onRequest((request, response) => {
    const agent = new WebhookClient({ request, response });
    console.log('Dialogflow Request headers: ' + JSON.stringify(request.headers));
    console.log('Dialogflow Request body: ' + JSON.stringify(request.body));

    async function countstuff(agent) {
        try {
            let msg = { 'type': 'count', 'request': request.body }
            const dataBuffer = Buffer.from(JSON.stringify(msg));
            const connection = new Connection('amqp://user:aaa@35.222.19.94')
            await connection.init()
            const channel = await connection.createChannel()
            const respChannel = await connection.createChannel()
            await channel.assertQueue('dialogflow_request')
            await respChannel.assertQueue('dialogflow_response')
            await respChannel.prefetch(1);
            let respMsg = await respChannel.get('dialogflow_response');
            while(respMsg) {
                respChannel.ack(respMsg);
                respMsg = await respChannel.get('dialogflow_response');
            }
            await channel.sendToQueue('dialogflow_request', Buffer.from(JSON.stringify(msg)))
            
            
            respMsg = false
            while(!respMsg) {
                await respChannel.prefetch(1);
                respMsg = await respChannel.get('dialogflow_response');
            }
            agent.add(respMsg.content.toString());
            console.log(respMsg.content.toString())
            respChannel.ack(respMsg);
            await respChannel.close()
            await channel.close()
            await connection.close()

        } catch (err) {
            console.log(err);
            agent.add("I am Sorry but I am not able to talk to the camera. Try again later.");
            console.log("error added");
        }
        console.log('done');

    }


    let intentMap = new Map();
    intentMap.set('count gummies', countstuff);
    // intentMap.set('your intent name here', googleAssistantHandler);
    agent.handleRequest(intentMap);
});
