package com.a504.chatcomposer.global.config;

import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.connection.CachingConnectionFactory;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitMQConfig {
    @Value("${spring.rabbitmq.host}")
    private String host;

    @Value("${spring.rabbitmq.username}")
    private String username;

    @Value("${spring.rabbitmq.password}")
    private String password;

    @Value("${spring.rabbitmq.port}")
    private int port;

    @Bean
    Queue queue() {
        return new Queue("response.queue", true);
    }

    @Bean
    DirectExchange directExchange() {
        return new DirectExchange("music.exchange");
    }

    @Bean
    FanoutExchange fanoutExchange() {
        return new FanoutExchange("music.fanout");
    }
//    @Bean
//    Binding binding(DirectExchange directExchange, Queue queue) {
////        return BindingBuilder.bind(queue).to(directExchange).with("diffusion.key");
//        return BindingBuilder.bind(queue).to(directExchange).with();
//    }

    @Bean
    Binding binding(FanoutExchange fanoutExchange, Queue queue) {
//        return BindingBuilder.bind(queue).to(directExchange).with("diffusion.key");
        return BindingBuilder.bind(queue).to(fanoutExchange);
    }

    @Bean
    RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory, MessageConverter messageConverter) {
        RabbitTemplate rabbitTemplate = new RabbitTemplate(connectionFactory);
        rabbitTemplate.setMessageConverter(messageConverter);
        rabbitTemplate.setReplyTimeout(100000); // 100 seconds in milliseconds
        return rabbitTemplate;
    }

    @Bean
    ConnectionFactory connectionFactory() {
        CachingConnectionFactory connectionFactory = new CachingConnectionFactory();
        connectionFactory.setHost(host);
        connectionFactory.setPort(port);
        connectionFactory.setUsername(username);
        connectionFactory.setPassword(password);
        return connectionFactory;
    }

    @Bean
    MessageConverter messageConverter() {
        return new Jackson2JsonMessageConverter();
    }
}
