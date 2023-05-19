package com.a504.chatcomposer.user.repository;

import com.a504.chatcomposer.user.entity.QUser;
import com.querydsl.jpa.impl.JPAQueryFactory;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * User 관련 디비 쿼리를 구현한 클래스 (Querydsl 사용)
 */
@RequiredArgsConstructor
@Slf4j
public class UserRepositoryImpl implements UserCustomRepository {

	private final JPAQueryFactory jpaQueryFactory;
	QUser user = QUser.user;

	@Override
	public Long findUserSeqByUserId(String userId) {

		return jpaQueryFactory
			.select(user.userSeq)
			.from(user)
			.where(user.userId.eq(userId))
			.fetchOne();
	}
}
