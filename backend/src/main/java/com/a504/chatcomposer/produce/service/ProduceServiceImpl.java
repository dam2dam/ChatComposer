package com.a504.chatcomposer.produce.service;

import com.a504.chatcomposer.global.util.S3Uploader;
import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import com.a504.chatcomposer.produce.dto.response.CoverUrlResp;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.AmqpException;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.core.MessagePostProcessor;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

@Service("ProduceService")
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ProduceServiceImpl implements ProduceService {
    private final Logger LOGGER = LoggerFactory.getLogger(ProduceServiceImpl.class);
    private final S3Uploader s3Uploader;
    private final RabbitTemplate rabbitTemplate;

//    @Override
//    public CoverUrlResp createCover(String coverRequest) throws JsonProcessingException {
//        String url = "http://127.0.0.1:8885/api/generate?prompt=" + coverRequest;
//
//        // FastApi 외부통신으로 이미지를 가져온다.
//        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
//            HttpGet request = new HttpGet(url);
//
//            HttpResponse response = httpClient.execute(request);
//            HttpEntity entity = response.getEntity();
//
//            return EntityUtils.toByteArray(entity);
//
//        } catch (IOException e) {
//            // handle exception
//            throw new IllegalArgumentException("앨범 커버 생성에 실패했습니다.");
//        }
//        CoverUrlResp coverUrlResp = CoverUrlResp.of("성공!", coverRequest);
//
//        // correlation ID 생성 및 callback queue 설정
//        String correlationId = UUID.randomUUID().toString();
//        MessageProperties messageProperties = new MessageProperties();
//        messageProperties.setCorrelationId(correlationId);
//        messageProperties.setReplyTo("test.queue");
//
//        // 보낼 객체 Message 객체에 같이 보내기 위해 바이트 배열로 직렬화
//        ObjectMapper objectMapper = new ObjectMapper();
//        byte[] body = objectMapper.writeValueAsBytes(coverUrlResp);
//
//        // request queue에 보낼 message 설정
//        Message message = new Message(body, messageProperties);
//        rabbitTemplate.convertsendand("music.exchange", "diffusion.key", message);
//
//        // Wait for the reply message with the same correlation ID
//        Message replyMessage = rabbitTemplate.receive("test.queue", 5000L);
//        if (replyMessage == null) {
//            throw new IllegalArgumentException("RabbitMQ에서 받아온 message가 존재하지않습니다.");
//        }
//
//        return coverUrlResp;
//    }

    @Override
    public CoverUrlResp createCover(String coverRequest) throws IOException {
        CoverUrlResp coverUrlResp = CoverUrlResp.of("성공!", 200, coverRequest);

        // correlation ID 생성
        String correlationId = UUID.randomUUID().toString();

        // 보낼 객체 Message 객체에 같이 보내기 위해 바이트 배열로 직렬화
        ObjectMapper objectMapper = new ObjectMapper();
        byte[] body = objectMapper.writeValueAsBytes(coverUrlResp);

        // request queue에 보낼 message 설정
        MessageProperties messageProperties = new MessageProperties();
        messageProperties.setCorrelationId(correlationId);
        messageProperties.setReplyTo("response.queue");
//        messageProperties.setExpiration("1000000");
        Message message = new Message(body, messageProperties);

        // convertSendAndReceive() 함수를 사용하여 RPC 구현
        Message replyMessage = (Message) rabbitTemplate.convertSendAndReceive("music.exchange", "diffusion.key", message,
                new MessagePostProcessor() {
                    @Override
                    public Message postProcessMessage(Message message) throws AmqpException {
                        message.getMessageProperties().setCorrelationId(correlationId);
                        message.getMessageProperties().setReplyTo("response.queue");
//                        message.getMessageProperties().setExpiration("1000000"); // set the message expiration time to 10 seconds
                        return message;
                    }
                });

        if (replyMessage == null) {
            throw new IllegalArgumentException("RabbitMQ에서 받아온 message가 존재하지않습니다.");
        }
        // response message에서 결과 추출
        byte[] replyBody = replyMessage.getBody();
        CoverUrlResp result = objectMapper.readValue(replyBody, CoverUrlResp.class);

        return result;
    }

    @Transactional
    @Override
    public String saveCover(MultipartFileReq multipartFileReq) {
        // S3에 이미지 등록
        MultipartFile multipartFile = multipartFileReq.getImage();
        System.out.println(multipartFile.getOriginalFilename()+"11111111111111");
        String img;
        try {
            img = s3Uploader.upload(multipartFile, "album");
        } catch (IOException e) {
            throw new IllegalArgumentException("앨범 커버 저장에 실패했습니다.");
        }
        LOGGER.info("================url===============\n" + img);
        return img;
    }
}
