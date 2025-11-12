INSERT INTO public."user" (id,username,"password",name,avatar_url,email,phone_number) VALUES
	 ('0a852eab-dff7-4e0b-bc99-4edd7d13aa4d'::uuid,'admin','$2a$12$pj2bBHJdYqvKTLoIkChv7eGGiQm/VbvrcJ9NpQ3vRfgPKpvzrxzei','hiimadmin','example_url','admin@gmail.com','982359246'),
	 ('94811f40-fada-4d84-a1e3-cf1c5131f1a6'::uuid,'minh','$2b$12$zn.0mRrLOTbU6QYsFvJeP.25SHjgdbpebp/rzHRv7CjI31QLrYCYK','minh','minh_image','minhtb03@gmail.com','823475698')
ON CONFLICT (id) DO NOTHING;

INSERT INTO public."role" (id,name) VALUES
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'ADMIN'),
	 ('26c9db41-4fc4-4b10-9101-42bf124e66a6'::uuid,'USER')
ON CONFLICT (id) DO NOTHING;

INSERT INTO public."permission" (id,name) VALUES
	 ('1d8b5eca-6943-4046-a1ef-f43b92d58213'::uuid,'READ_USERS'),
	 ('b6566ffe-33d2-4f77-a444-f605e2c443e8'::uuid,'CREATE_USER'),
	 ('c205a222-a972-49d6-9a3c-f4021dde35c8'::uuid,'READ_USER'),
	 ('8441f94c-cac3-46fa-a506-d685b0c5aa00'::uuid,'UPDATE_USER'),
	 ('e03b4154-1159-449f-afff-b3a112bb1d71'::uuid,'DELETE_USER'),
	 ('37868ad0-6b41-48ff-a477-eae8d53f2b38'::uuid,'READ_ROLES'),
	 ('372a5a11-7970-4398-a5d7-aa306f1357aa'::uuid,'CREATE_ROLE'),
	 ('7dbfff5a-afa7-4956-96eb-d32922cfa23d'::uuid,'READ_ROLE'),
	 ('964bc32e-d210-4409-84e2-a8c36ab5d516'::uuid,'UPDATE_ROLE'),
	 ('ee95e823-dc3a-4d7b-bd70-7467a6e3c2f2'::uuid,'DELETE_ROLE'),
	 ('602d241c-99d5-4815-839a-ad9192c3cd87'::uuid,'APPROVE_REQUEST'),
	 ('4f05e87a-3aa5-4ec8-89ce-5c212a79dd6e'::uuid,'REJECT_REQUEST'),
	 ('7c2fd190-abe6-4886-bc8c-bfd6dd1cf819'::uuid,'INDEXING'),
	 ('f3621d43-0158-4202-8c0f-86e1010007c3'::uuid,'GEN_VIDEO'),
	 ('f6dae556-23e9-4d91-a387-30a26036d2b7'::uuid,'PREPARE_AVATAR_DATA'),
ON CONFLICT (id) DO NOTHING;

INSERT INTO public.user_role (user_id,role_id) VALUES
	 ('0a852eab-dff7-4e0b-bc99-4edd7d13aa4d'::uuid,'419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid),
	 ('94811f40-fada-4d84-a1e3-cf1c5131f1a6'::uuid,'26c9db41-4fc4-4b10-9101-42bf124e66a6'::uuid)
ON CONFLICT (user_id, role_id) DO NOTHING;

INSERT INTO public.role_permission (role_id,permission_id) VALUES
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'1d8b5eca-6943-4046-a1ef-f43b92d58213'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'b6566ffe-33d2-4f77-a444-f605e2c443e8'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'c205a222-a972-49d6-9a3c-f4021dde35c8'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'8441f94c-cac3-46fa-a506-d685b0c5aa00'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'e03b4154-1159-449f-afff-b3a112bb1d71'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'37868ad0-6b41-48ff-a477-eae8d53f2b38'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'372a5a11-7970-4398-a5d7-aa306f1357aa'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'7dbfff5a-afa7-4956-96eb-d32922cfa23d'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'964bc32e-d210-4409-84e2-a8c36ab5d516'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'ee95e823-dc3a-4d7b-bd70-7467a6e3c2f2'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'602d241c-99d5-4815-839a-ad9192c3cd87'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'4f05e87a-3aa5-4ec8-89ce-5c212a79dd6e'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'7c2fd190-abe6-4886-bc8c-bfd6dd1cf819'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'f3621d43-0158-4202-8c0f-86e1010007c3'::uuid),
	 ('419935b9-3c66-47ad-9dd2-874ddf0f7551'::uuid,'f6dae556-23e9-4d91-a387-30a26036d2b7'::uuid),
ON CONFLICT (role_id, permission_id) DO NOTHING;
