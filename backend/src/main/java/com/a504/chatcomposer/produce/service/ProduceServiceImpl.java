package com.a504.chatcomposer.produce.service;

import com.a504.chatcomposer.global.util.S3Uploader;
import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import com.a504.chatcomposer.produce.dto.response.FileUrlResp;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import com.rabbitmq.client.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

@Service("ProduceService")
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class ProduceServiceImpl implements ProduceService {
    private final Logger LOGGER = LoggerFactory.getLogger(ProduceServiceImpl.class);
    private final S3Uploader s3Uploader;
    private final Channel channel;
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
    public FileUrlResp createCover(String coverRequest) throws IOException, InterruptedException {
        FileUrlResp fileUrlResp = FileUrlResp.of("Success!", 200, coverRequest);

        // Generate correlation ID
        String correlationId = UUID.randomUUID().toString();

        // Convert CoverUrlResp object to byte array
        ObjectMapper objectMapper = new ObjectMapper();
        byte[] body = objectMapper.writeValueAsBytes(fileUrlResp);

        //unique Queue create
//        String replyQueueName = channel.queueDeclare().getQueue();
        String replyQueueName = "response.queue";

        //properties setting
        AMQP.BasicProperties props = new AMQP.BasicProperties
                .Builder()
                .correlationId(correlationId)
                .replyTo(replyQueueName)
                .build();

        //Send Request
        channel.basicPublish("music.exchange", "diffusion.key", props, body);

        //차단하는 Queue...? 추측: correlationId 가 다르면 차단
        final BlockingQueue<String> response = new ArrayBlockingQueue<>(1);

        DeliverCallback callback = (consumerTag, delivery) -> {
            //되받은 메세지에 담긴 correlationId를 요청보낸 correlationId와 대조
            if(delivery.getProperties().getCorrelationId().equals(correlationId))
                response.offer(new String(delivery.getBody(), "UTF-8"));
        };

        String ctag = channel.basicConsume(replyQueueName, true, callback, consumerTag -> {});
        String tmp = response.take();
        channel.basicCancel(ctag);

        // extract result from response message
//        byte[] replyBody = replyMessage.getBody();
        FileUrlResp result = FileUrlResp.of("Success!", 200, tmp);

        return result;
    }

    @Transactional
    @Override
    public String saveCover(MultipartFileReq image) {
        // S3에 이미지 등록
        MultipartFile multipartFile = image.getFile();
        String img;
        try {
            img = s3Uploader.upload(multipartFile, "album");
        } catch (IOException e) {
            throw new IllegalArgumentException("앨범 커버 저장에 실패했습니다.");
        }
        LOGGER.info("================url===============\n" + img);
        return img;
    }

    @Override
    public String saveMIDI(MultipartFileReq file) {
        // S3에 이미지 등록
        MultipartFile multipartFile = file.getFile();
        String img;
        try {
            img = s3Uploader.upload(multipartFile, "origin-music");
        } catch (IOException e) {
            throw new IllegalArgumentException("원본 음악 저장에 실패했습니다.");
        }
        LOGGER.info("================url===============\n" + img);
        return img;
    }
}
