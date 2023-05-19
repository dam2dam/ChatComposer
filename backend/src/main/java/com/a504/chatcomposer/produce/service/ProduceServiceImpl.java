package com.a504.chatcomposer.produce.service;

import com.a504.chatcomposer.global.util.S3Uploader;
import com.a504.chatcomposer.produce.dto.request.MixedMusicReq;
import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import com.a504.chatcomposer.produce.dto.request.OriginalMusicReq;
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

    @Override
    public byte[] createMusic(String musicSource, String prompt) throws IOException, InterruptedException {
        // Generate correlation ID
        String correlationId = UUID.randomUUID().toString();

        // Convert CoverUrlResp object to byte array
        ObjectMapper objectMapper = new ObjectMapper();
        byte[] body = objectMapper.writeValueAsBytes(MixedMusicReq.of(musicSource, prompt));

        //unique Queue create
        String replyQueueName = "response.queue";

        //properties setting
        AMQP.BasicProperties props = new AMQP.BasicProperties
                .Builder()
                .correlationId(correlationId)
                .replyTo(replyQueueName)
                .build();

        //Send Request
        channel.basicPublish("music.exchange", "riffusion.key", props, body);

        //차단하는 Queue... correlationId 이 다르면 차단
        final BlockingQueue<byte[]> response = new ArrayBlockingQueue<>(1);

        DeliverCallback callback = (consumerTag, delivery) -> {
            //되받은 메세지에 담긴 correlationId를 요청보낸 correlationId와 대조
            if(delivery.getProperties().getCorrelationId().equals(correlationId)){
                response.offer(delivery.getBody());
            }
        };

        String ctag = channel.basicConsume(replyQueueName, true, callback, consumerTag -> {});
        byte[] originalMusic = response.take();
        channel.basicCancel(ctag);
        
        return originalMusic;
    }

    @Override
    public byte[] createCover(String coverRequest) throws IOException, InterruptedException {
        // Generate correlation ID
        String correlationId = UUID.randomUUID().toString();

        // Convert coverRequest object to byte array
        byte[] body = coverRequest.getBytes("UTF-8");

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
        final BlockingQueue<byte[]> response = new ArrayBlockingQueue<>(1);

        DeliverCallback callback = (consumerTag, delivery) -> {
            //되받은 메세지에 담긴 correlationId를 요청보낸 correlationId와 대조
            if(delivery.getProperties().getCorrelationId().equals(correlationId)){
                LOGGER.info("\n====callback====\n");
                response.offer(delivery.getBody());
            }

        };

        String ctag = channel.basicConsume(replyQueueName, true, callback, consumerTag -> {});
        byte[] result = response.take();
        channel.basicCancel(ctag);

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

    @Transactional
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

    @Transactional
    @Override
    public String saveMusic(MultipartFileReq file) {
        // S3에 이미지 등록
        MultipartFile multipartFile = file.getFile();
        String music;
        try {
            music = s3Uploader.upload(multipartFile, "mixed-music");
        } catch (IOException e) {
            throw new IllegalArgumentException("믹싱 음악 저장에 실패했습니다.");
        }
        LOGGER.info("================url===============\n" + music);
        return music;
    }
}
