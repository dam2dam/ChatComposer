package com.a504.chatcomposer.user.repository;

import org.springframework.stereotype.Repository;

/**
 * User 관련 디비 쿼리를 정의한 인터페이스 (Querydsl 사용)
 */
@Repository
public interface UserCustomRepository {

	Long findUserSeqByUserId(String userId);
}
